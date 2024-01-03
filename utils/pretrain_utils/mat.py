import re
import torch

def remove_consecutive_lines(code):
    if code[-1] != '\n':
        code += '\n'
    # Del line indices start from 0
    del_line_indices = []
    new_line_indices = []
    new_code = ''
    code_ptr = 0
    for m in re.finditer('\n', code):  # 3
        new_line_indices.append(m.start())
    if len(new_line_indices) > 0:
        for i in range(len(new_line_indices) - 1):
            # Consecutive lines
            # if new_line_indices[i] == new_line_indices[i + 1] + 1:
            if code[new_line_indices[i]:new_line_indices[i + 1] + 1].strip() == '':
                del_line_indices.append(i + 1)
                new_code += code[code_ptr:new_line_indices[i]+1]
                code_ptr = new_line_indices[i + 1] + 1
            else:
                new_code += code[code_ptr:new_line_indices[i]+1]
                code_ptr = new_line_indices[i]+1

    new_code += code[code_ptr:]

    # while '\n\n' in code:
    #     code = code.replace('\n\n', '\n')
    return new_code, del_line_indices

def shift_graph_matrix(mat, del_lines, shift=0):
    assert len(mat.shape) == 2 and mat.size(0) == mat.size(1)
    line_spans = [-1] + [n-shift for n in del_lines] + [len(mat)]
    # print(line_spans)

    new_rows = []
    for start, end in zip(line_spans[:-1],line_spans[1:]):
        if start < end:
            new_rows.append(mat[start+1:end])
    new_mat = torch.cat(new_rows, dim=0)

    new_cols = []
    for start, end in zip(line_spans[:-1],line_spans[1:]):
        if start < end:
            new_cols.append(new_mat[:, start+1:end])
    new_mat = torch.cat(new_cols, dim=1)
    return new_mat

def shift_edges_in_matrix(edges, del_line_indices, shift=-1):
    """
        This method is used to shift the edges in a matrix after certain lines
        were removed.
        Edges are input as compressed form, rather than matrix form.

        E.g.:
        With Del Lines: [3]   ,
        Edges: [(1,2),(1,4),(4,5)]    →     Shifted edges: [(1,2),(1,3),(3,4)]

        Demostration：
          1 2 3 4 5                            1 2   3 4
        [[0,1,0,1,0],  1                     [[0,1,×,1,0],      1
         [0,0,0,0,0],  2                      [0,0,×,0,0],      2
         [0,0,0,0,0],  3             →        [×,×,×,×,×],
         [0,0,0,0,1],  4                      [0,0,×,0,1],      3
         [0,0,0,0,0]]  5                      [0,0,×,0,0]]      4

        Note: Shift=-1 works for edges start from 1 but del_indices start from 0.
    """
    if len(edges) == 0:
        return edges
    edges_t = torch.LongTensor(edges)
    e_max = int(edges_t.max().item())
    temp_mat = torch.zeros((e_max+1, e_max+1))
    temp_mat[edges_t[:,0], edges_t[:,1]] = 1
    # Make up start from 0 and start from 1
    temp_mat = shift_graph_matrix(temp_mat, del_line_indices, shift)
    return temp_mat.nonzero().tolist()