import dgl

def get_kg(kg_df):
    node1 = kg_df['node1'].to_numpy()
    node2 = kg_df['node2'].to_numpy()
    interaction = kg_df['interaction'].to_numpy()

    edge_types = list(set(interaction))

    data_dict = {}
    for edge_type in edge_types:
        edge_indices = (interaction == edge_type)
        src_nodes = node1[edge_indices]
        dst_nodes = node2[edge_indices]
        data_dict[('node', f'rel_{edge_type}', 'node')] = (src_nodes, dst_nodes)

    hg = dgl.heterograph(data_dict)

    return hg