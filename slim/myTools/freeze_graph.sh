python3 -m tensorflow.python.tools.freeze_graph --input_graph=graph.pbtxt --input_checkpoint=model.ckpt-1000.data-00000-of-00001 --output_node_names=fc4 --output_graph=out.pb

