## 一. 基本流程                                                                                                            
1.通过解析，把任意框架的图结构，转换为预定义的GraphDef结构来表示，权重转换为dict()表示. 这种已转换的统一结构，称之为中间表示.

2.中间表示只是一种序列化的结构，定义在graph.proto中，使用GraphDef表示图，NodeDef表示点. 这个序列化结构非常类似tensorflow所使用的图结构.

3.这样在解析特定框架的时候，需要加载特定的图结构和权重：
对于tensorflow来说，使用TensorflowGraph来表示自己的图，TensorflowGraphNode来表示图中的节点.通过ParseFromString(file_content)加载图.
对于pytorch来说，使用PytorchGraph来表示自己的图，PytorchGraph来表示图中的节点.通过torch.load(model_file_name)加载图.

4.尽管各个框架具体的图定义都不同，但是都有一些共性，因此TensorflowGraph和PytorchGraph都继承了共同的基类Graph.
同时，用于中间表示的图IRGraph也是继承于Graph.(IRGraph于GraphDef的关系：IRGraph包裹GraphDef.)

## 二. 存储结构
类似于存储有向图的 **十字链表** 结构。

1.使用OrderedMap存储所有结点（即顶点），结点名称作为键；而十字链表使用数组存储所有结点，以数组下标区分不同结点。

2.结点结构中，有两个类似链表的列表：in_nodes_names和out_nodes_names，来表示结点之间的有向关系；十字链表中使用hlink和tlink来链接具有相同依赖关系的所有结点。

## 三. 拓扑排序
经过拓扑排序，把有向无环图中所有结点的偏序关系，输出为全序关系。这样就可以顺序处理结点以转换为目标框架模型。

依次找到入度为0的结点，输出并删除该结点，然后反复该过程，直到所有结点都输出。
初始结点即输入结点。