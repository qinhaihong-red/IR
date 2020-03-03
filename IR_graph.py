#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.utils import *
from mmdnn.conversion.common.IR.graph_pb2 import TensorShape, AttrValue
from mmdnn.conversion.common.DataStructure.graph import Graph, GraphNode

#container即model
def load_protobuf_from_file(graph_model, filename):
    with open(filename, 'rb') as f:
        file_content = f.read()

    # First try to read it as a binary file.
    try:
        graph_model.ParseFromString(file_content)
        print("Parse file [%s] with binary format successfully." % (filename))
        return graph_model

    except Exception as e:  # pylint: disable=broad-except
        print ("Info: Trying to parse file [%s] with binary format but failed with error [%s]." % (filename, str(e)))

    # Next try to read it as a text file.
    try:
        from google.protobuf import text_format
        text_format.Parse(file_content.decode('UTF-8'), graph_model, allow_unknown_extension=True)
        print("Parse file [%s] with text format successfully." % (filename))
    except text_format.ParseError as e:
        raise IOError("Cannot parse file %s: %s." % (filename, str(e)))

    return graph_model

class IRGraphNode(GraphNode):

    @staticmethod
    def replace_scope(name):
        return name.replace('/', '_')

    @property
    def IR_node(self):
        return self.node 

    @property
    def name(self):
        return self.node.name #name属性定义在GraphNode中. 3个顶级属性之一.

    @property
    def type(self):
        return self.node.op #即操作名.3个顶级属性之一.

    def set_attrs(self, attrs):
        assign_IRnode_values(self.node, attrs)#多个属性，每个属性只存储一个类型的值

    #具体可见graphdef_snippet.py的示例
    def get_attr(self, name, default_value = None):
        if name in self.node.attr:
            attr = self.node.attr[name]
            field = attr.WhichOneof('value')
            val = getattr(attr, field) if field else default_value
            if not val:
                return val
            if isinstance(val, AttrValue.ListValue):
                if val.ListFields():
                    return list(val.ListFields()[0][1])
                else:
                    return val.ListFields()
            else:
                return val.decode('utf-8') if isinstance(val, bytes) else val
        else:
            return default_value


#包裹中间表示模型GraphDef，用于具体框架的Emitter生成该框架的专用代码和权重文件.
#而具体的图结构，如tensorflow_graph和pytorch_graph，都是继承自Graph，与IRGraph和IRGraphNode没啥关系.

class IRGraph(Graph):
    
    #在parser中统一使用GraphDef定义中间表示模型. 这里读出后，再使用IRGraph包裹这个模型.
    def __init__(self, filename):
        model = graph_pb2.GraphDef()
        load_protobuf_from_file(model, filename) #从文件中读取中间表示模型
        super(IRGraph, self).__init__(model)


    #应该是过滤掉输入边和输出边为0的节点
    def filter_node(self):
        self.node_map = dict(filter(lambda node: node[1].in_nodes_names or node[1].out_nodes_names, self.node_map.items()))

    #展开中间表示模型
    def build(self):
        for node in self.model.node:#遍历GraphDef的所有node，即NodeDef
            self.node_map[node.name] = IRGraphNode(node)#包裹节点(层).在基类GraphNode中初始化.
            self.node_name_map[node.name] = node.name

        for _, node in enumerate(self.model.node):
            for pred_node_names in node.input:#node.input是NodeDef的3个顶级属性，以 string:list 表示的输入节点名.
                self._make_connection(pred_node_names, node.name)

        self.filter_node()
        super(IRGraph, self).build()
        #下面的.type()并非node的顶级属性，但是通过IRGraphNode的type()属性，返回的node.op
        self.input_nodes_names = list(filter(lambda node_name: self.node_map[node_name].type != 'Constant', self.input_nodes_names))


    def rebuild(self):
        self.input_nodes_names.clear()
        self.output_nodes_names.clear()
        self.topological_sorted_nodes_names.clear()
        self.filter_node()
        super(IRGraph, self).build()
        #下面的.type()并非node的顶级属性，但是通过IRGraphNode的type()属性，返回的node.op
        self.input_nodes_names = list(filter(lambda node_name: self.node_map[node_name].type != 'Constant', self.input_nodes_names))


    def clear_out_scope_node(self):

        def _clear_names_out_scope(names):
            for idx in range(len(names) -1, -1, -1):
                node = self.get_node_by_name(names[idx])
                #这个.type通过IRGraphNode的type()属性，返回的node.op
                if node.type != 'Scope' and node.get_attr('scope'):
                    del names[idx]

        _clear_names_out_scope(self.input_nodes_names)
        _clear_names_out_scope(self.topological_sorted_nodes_names)
        _clear_names_out_scope(self.output_nodes_names)

    @staticmethod
    def shapeToStr(tensor_shape, keep_minus_one = False):
        ret = ""
        first = True
        for dim in tensor_shape.dim:
            if dim.size != -1 or keep_minus_one:
                if first == False:
                    ret += ", "
                ret += str(dim.size)
                first = False
        return ret
