#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import numpy as np
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType


#解析指定框架的图结构和权重文件，并生成中间表示GraphDef(图结构+权重).
class Parser(object):

    node_index=0

    def __init__(self):
        #图的中间表示：根据src_graph, 统一使用GraphDef，来构造图的中间表示并生成.pb. 然后在具体的Emitter中使用这个中间表示（通过IRGraph包裹），来生成其它指定框架的代码和权重文件.        
        self.IR_graph = GraphDef()#GraphDef的构造过程(.node.add())，见于具体的parser子类实现
        self.weight_loaded = False

        # name --> (weight_name --> ndarray)
        self.nodes_weights = dict()#权重的中间表示：以dict为元素的dict


    def run(self, dest_path):
        self.gen_IR()#生成中间表示
        print('save files turned off.')
        self.save_to_json(dest_path + ".json")
        # self.save_to_proto(dest_path + ".pb")
        self.save_weights(dest_path + ".npy")
        self.save_weights_txt(dest_path + ".txt")
    


    @property
    def src_graph(self):#图的原始表示：特定于框架. 比如tensorflow_parser的src_graph就是：self.tf_graph = TensorflowGraph(model) ,而model又是从.ckpt.meta中读出来的GraphDef
        raise NotImplementedError 


    def get_node_son_by_name(self, name, path, set_flag = False):
        return self.src_graph.get_node_son_by_name(name, path, set_flag)


    def get_node_parent_by_name(self, name, path, set_flag = False):
        return self.src_graph.get_node_parent_by_name(name, path, set_flag)

    
    def set_node_weight(self, node_name, weight_name, data):
        if node_name not in self.nodes_weights:
            self.nodes_weights[node_name] = dict()#以dict为元素的dict
            self.nodes_weights[node_name]['index']=self.node_index
            self.node_index+=1
        node_weight = self.nodes_weights[node_name]
        node_weight[weight_name] = data 


    #IR_graph本来就是GraphDef对象，因此可以直接序列化
    def save_to_json(self, filename):
        import google.protobuf.json_format as json_format
        json_str = json_format.MessageToJson(self.IR_graph, preserving_proto_field_name = True)#中间表示序列化为.json

        with open(filename, "w") as of:
            of.write(json_str)

        print ("IR network structure is saved as [{}].".format(filename))

        return json_str


    def save_to_proto(self, filename):
        proto_str = self.IR_graph.SerializeToString()#中间表示序列化为.pb
        with open(filename, 'wb') as of:
            of.write(proto_str)

        print ("IR network structure is saved as [{}].".format(filename))

        return proto_str


    def save_weights(self, filename):
        if self.weight_loaded:
            with open(filename, 'wb') as of:
                np.save(of, self.nodes_weights)#weights就是dict(), key:节点名, val:权重dict. 保存到.npy供Emitter进一步转换为特定框架的权重.
            print ("IR weights are saved as [{}].".format(filename))

        else:
            print ("Warning: weights are not loaded.")

    def save_weights_txt(self, filename):
        if self.weight_loaded:
            with open(filename, 'w') as of:
                for name in self.nodes_weights:
                    of.write(name+':\n')
                    for name2 in self.nodes_weights[name]:
                        if name2=='index':
                            of.write('\t id:{}\n'.format(self.nodes_weights[name]['index']))
                        else:
                            of.write('\t {} shape:{} \n'.format(name2,self.nodes_weights[name][name2].shape))
            
            print ("IR weights are saved as [{}].".format(filename))

        else:
            print ("Warning: weights are not loaded.")

    #原名：convert_inedge 
    #注意这个方法在tensorflow_parser中没有用到.而在tensorflow_parser中定义了新函数_convert_in_nodes实现类似功能.
    def convert_in_nodes(self, src_node, IR_node, start_idx = 0, end_idx = None):
        if end_idx == None: 
            end_idx = len(src_node.in_nodes_names)
        
        for idx in range(start_idx, end_idx):
            #把node的输入关系，转换到中间表示NodeDef的input顶级属性中去.
            IR_node.input.append(self.src_graph.get_node_by_name(src_node.in_nodes_names[idx]).real_name.lstrip('_'))


    @staticmethod
    def channel_first_conv_kernel_to_IR(tensor):
        dim = tensor.ndim
        tensor = np.transpose(tensor, list(range(2, dim)) + [1, 0])
        return tensor


    @staticmethod
    def channel_first_shape_to_IR(shape):
        return [shape[0]] + list(shape[2:]) + [shape[1]]

    @staticmethod
    def channel_first_axis_to_IR(index):
        if index == 0:
            return 0
        elif index == 1:
            return -1
        else:
            return index - 1
