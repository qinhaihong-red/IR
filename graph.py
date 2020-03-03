from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import  collections

'''
基本流程：
1.通过解析，把任意框架的图结构，转换为预定义的GraphDef结构来表示，权重转换为dict()表示. 这种已转换的统一结构，称之为中间表示.

2.中间表示只是一种序列化的结构，定义在graph.proto中，使用GraphDef表示图，NodeDef表示点. 这个序列化结构非常类似tensorflow所使用的图结构.

3.这样在解析特定框架的时候，需要加载特定的图结构和权重：
对于tensorflow来说，使用TensorflowGraph来表示自己的图，TensorflowGraphNode来表示图中的节点.通过ParseFromString(file_content)加载图.
对于pytorch来说，使用PytorchGraph来表示自己的图，PytorchGraph来表示图中的节点.通过torch.load(model_file_name)加载图.

尽管各个框架具体的图定义都不同，但是都有一些共性，因此TensorflowGraph和PytorchGraph都继承了共同的基类Graph.
同时，用于中间表示的图IRGraph也是继承于Graph.(IRGraph于GraphDef的关系：IRGraph包裹GraphDef.)

'''

'''
存储结构：
类似于有向图的 十字链表 结构。
1.使用OrderedMap存储所有结点（即顶点），结点名称作为键；而十字链表使用数组存储所有结点，以数组下标区分不同结点。
2.结点结构中，有两个类似链表的列表：in_nodes_names和out_nodes_names，来表示结点之间的有向关系；十字链表中使用hlink和tlink来链接所有相同依赖关系的结点。

'''


'''
拓扑排序：
经过拓扑排序，把有向无环图中所有结点的偏序关系，输出为全序关系。
依次找到入度为0的结点，输出并删除该结点，然后反复该过程，直到所有结点都输出。
初始结点即输入结点。

'''

#GraphNode和Graph既是解析特定框架图的基类，也是包裹中间表示图的基类.


#GraphNode:
# 1.包裹NodeDef(对于中间表示IRGraphNode)
# 2.包裹指定框架图中的Node
class GraphNode(object):

    def __init__(self, node):
        self.node = node
        self.in_nodes_names = list()#建立节点之间的联系：输入该节点的节点名称 .根据上下文，这里的name是：op_name:out_index
        self.out_nodes_names = list()#建立节点之间的联系：从该节点输出指向的节点名称
        self.covered = False
        self.real_name = self.name

    @property
    def name(self):
        assert False

    @property
    def variable_name(self):
        return self.real_name.replace('/', '_').replace('-', '_').replace('[','_').replace(']','_')

    @property
    def real_variable_name(self):
        return self.real_name.replace('/', '_').replace('-', '_').replace('[','_').replace(']','_')


#Graph:
#1.Emitter中用IRGraph包裹中间表示的GraphDef
#2.TensorflowGraph/PyTorchGraph包裹具体源框架的model
class Graph(object):

    def __init__(self, model):
        self.model = model #对于源框架，这个model就是具体于各框架的模型；对于中间表示IRGraph，这个model就是由.pb文件ParseFromString加载出来的GraphDef.
        self.node_map = collections.OrderedDict()# NodeDef map. key为node.name, value为node. 根据上下文，这里的name只有:op_name，不带out_index.
        self.node_name_map = collections.OrderedDict()# 对于子类tensorflow_graph，这个node_name_map没有什么用. 因为node_name_map中的name就是node.name 


        self.input_nodes_names = list() #存放整个图的输入节点名称
        self.output_nodes_names = list()#存放整个图的输出节点名称
        self.topological_sorted_nodes_names = list()#存储图的拓扑关系

    def build(self):
        self._make_input_nodes()
        self._make_output_nodes()
        self._make_topological_sorted_nodes()

    def rebuild(self):
        self._make_input_nodes(True)
        self._make_output_nodes()
        self._make_topological_sorted_nodes()

    #遍历所有结点，找到输入结点
    def _make_input_nodes(self, rebuild=False):
        for name, node in self.node_map.items():
            node.left_in_nodes_count = len(node.in_nodes_names)#该节点剩余输入节点数量
            if len(node.in_nodes_names) == 0:#输入边为0的层作为输入层
                if rebuild:
                    if not node.get_attr('scope'):#get_attr属于子类方法
                        self.input_nodes_names.append(name)
                else:
                    self.input_nodes_names.append(name)

    #遍历所有结点，找到输出结点
    def _make_output_nodes(self):
        for name, node in self.node_map.items():
            if len(node.out_nodes_names) == 0:#输出边为0的层作为输出层
                self.output_nodes_names.append(name)


    '''get node by its name or tensor name'''
    def get_node_by_name(self, name):
        if not name.split(':')[0] in self.node_map:
            raise IOError("Graph doesn't have node [%s]." % name.split(':')[0])
            return None
        else:
            return self.node_map[name.split(':')[0]]#如果name为：op:0，那么name.split(':')[0]就能抽取出op


    def get_nodes(self):
        return self.node_map.values()

    #这个path是类似于[0,1,0]这么的一个index列表
    def get_node_son_by_name(self, name, path, set_flag = False):
        if name == None: return None
        current_node = self.get_node_by_name(name)
        for idx in path:
            if len(current_node.out_nodes_names) <= idx: return None
            son_name = current_node.out_nodes_names[idx].split(':')[0]
            current_node = self.get_node_by_name(son_name)
            if set_flag:
                current_node.covered = True
        return current_node

    #这个path是类似于[0,1,0]这么的一个index列表
    #比如图拓扑为：
    # a->r->x
    # b->r->x
    #    t->x
    #则有：
    # get_node_parent_by_name(x,[0])   -> r
    # get_node_parent_by_name(x,[1])   -> t
    # get_node_parent_by_name(x,[0,0]) -> a
    # get_node_parent_by_name(x,[0,1]) -> b
    
    def get_node_parent_by_name(self, name, path, set_flag = False):
        if name == None: return None
        current_node = self.get_node_by_name(name)
        for idx in path:
            if len(current_node.in_nodes_names) <= idx: return None
            parent_name = current_node.in_nodes_names[idx].split(':')[0]
            current_node = self.get_node_by_name(parent_name)
            if set_flag:
                current_node.covered = True
        return current_node

    def get_real_parent_name(self, name, path, set_flag = False):
        if name == None: return None
        current_node = self.get_node_by_name(name)
        for idx in path:
            if len(current_node.in_nodes_names) <= idx: return None
            parent_name = current_node.in_nodes_names[idx].split(':')[0]
            current_node = self.get_node_by_name(parent_name)
            if set_flag:
                current_node.covered = True
        return self.node_name_map[current_node.name]


    def get_parent_variable_name(self, name, path, set_flag = False):
        if name == None: return None
        current_node = self.get_node_by_name(name)
        for idx in path:
            if len(current_node.in_nodes_names) <= idx: return None
            parent_name = current_node.in_nodes_names[idx].split(':')[0]
            current_subscriptor = '' if len(current_node.in_nodes_names[idx].split(':'))==1 else '[{}]'.format(current_node.in_nodes_names[idx].split(':')[1])
            current_node = self.get_node_by_name(parent_name)
            if set_flag:
                current_node.covered = True

        return current_node.real_variable_name + current_subscriptor


    # 按输入顺序和数量构造所有节点的拓扑序列
    def _make_topological_sorted_nodes(self):
        self.topological_sorted_nodes_names = self.input_nodes_names[:]
        idx = 0
        while idx < len(self.topological_sorted_nodes_names):
            current_node = self.get_node_by_name(self.topological_sorted_nodes_names[idx])
            for next_node_name in current_node.out_nodes_names:
                next_node = self.get_node_by_name(next_node_name)
                #只有遍历完所有与下个节点产生连接的节点之后，才能把下个节点添加到拓扑序列
                #例如：
                #   a-->x
                #   a-->y
                #   b-->y
                #遍历到a.out_nodes_name==x时，因为只有a与之产生联系，因此x.left_in_nodes_count（1）- 1 = 0. 因此添加到拓扑序列.
                #遍历到a.out_nodes_name==y时，同时有a,b与之产生联系，因此y.left_in_nodes_count（2）- 1 = 1. 因此暂时不添加到拓扑序列.
                #遍历到b.out_nodes_name==y时，y.left_in_nodes_count（1）- 1 = 0. 这时就可以添加到拓扑序列了.
                #这样，就把所有的节点，按输入顺序和数量，进行了拓扑排序.
                next_node.left_in_nodes_count -= self._check_left_in_nodes_num(current_node.name, next_node) # one node may connect another node by more than one edge. 
                # next_node_info.left_in_edges -= 1
                if next_node.left_in_nodes_count == 0:
                    self.topological_sorted_nodes_names.append(next_node_name)
            idx += 1


    #建立两个节点之间的输入、输出关系
    def _make_connection(self, src_node_name, dst_node_name):
        if (src_node_name == dst_node_name) or (src_node_name not in self.node_map) or (dst_node_name not in self.node_map):
            if src_node_name.split(':')[0] not in self.node_map:
                print ("Warning: Graph Construct a self-loop node {}. Ignored.".format(src_node_name))
                return

        # print ('{} --> {}'.format(src_node_name, dst_node_name))
        if dst_node_name not in self.node_map[src_node_name.split(':')[0]].out_nodes_names:
            self.node_map[src_node_name.split(':')[0]].out_nodes_names.append(dst_node_name)

        if src_node_name not in self.node_map[dst_node_name].in_nodes_names:
            self.node_map[dst_node_name.split(':')[0]].in_nodes_names.append(src_node_name)


    def _check_left_in_nodes_num(self, in_node_name, node):
        count = 0
        for name in node.in_nodes_names:
            if in_node_name == name.split(':')[0]:
                count += 1
        return count