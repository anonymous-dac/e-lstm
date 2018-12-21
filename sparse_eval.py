# Generate eSELL sparse matrix and compare with CSR, CSC, COO representation.

from scipy import sparse
import numpy as np
import sys
import math


def frac2fix(nd_array, dtype):
    dt_bit_len = np.dtype(dtype).itemsize * 8
    # mult_factor = np.array(2**dt_bit_len).repeat(nd_array.shape)
    nd_array = nd_array * (2 ** dt_bit_len)
    nd_array_int = nd_array.astype(np.dtype(dtype))
    return nd_array_int


def esell_construct(mat_in, idx_encodebook, chk_h, n_chk, chk_w):
    '''
    :param mat_in: input matrix
    :param idx_encodebook: encodebook for col_idx encoding
    :param chk_h: chunk size (CHK_h), note: val_c should be an integer multiple of 4, to avoid the padding overhead
    :param n_chk: ration of BLK_h/CHK_h
    :param chk_w: CHK_W/BLK_W (should be an integer multiple of 4)
    :return: mat_esell, mat_permut, mat_chkw, mat_col_idx_encode, mat_chk_overhead
    '''
    # obtain the matrix size
    (row, col) = mat_in.shape

    mat_esell = np.zeros([row, col], dtype='int16')

    # sorting range size (BLK_h, sigma), note: should be an integer multiple of chunk height
    blk_h = n_chk * chk_h

    # row permutation index
    mat_permut = np.zeros([row, int(col / chk_w)], dtype='int16')
    # nnz of each block row, (after permutated)
    mat_nnz_permut = np.zeros([row, int(col / chk_w)], dtype='int16')
    # col index
    mat_col_idx = np.zeros([row, col], dtype='int16')
    mat_col_idx_permut = np.zeros([row, col], dtype='int16')
    # col index encode
    mat_col_idx_encode = np.zeros([row, int(col / chk_w)], dtype='int16')

    # chk_w
    mat_chkw = np.zeros([int(row / chk_h), int(col / chk_w)], dtype='int16')
    mat_chk_overhead = np.zeros([row, int(col / chk_w)], dtype='int16')

    # mat slicing and sorting
    for idx_cw in range(0, int(col / chk_w)):
        for idx_blk in range(0, int(row / blk_h)):
            mat_slice = mat_in[idx_blk * blk_h:(idx_blk + 1) * blk_h, idx_cw * chk_w:(idx_cw + 1) * chk_w]
            # count and sort nnz
            vec_nnz = np.zeros(blk_h, dtype='int16')
            # iterate all line-elements of a block
            for t_blk in range(0, blk_h):
                nnz = np.count_nonzero(mat_slice[t_blk, :])
                vec_nnz[t_blk] = nnz
                # fill in the col idx
                if (nnz != 0):
                    mat_col_idx[idx_blk * blk_h + t_blk, idx_cw * chk_w:idx_cw * chk_w + nnz] = np.nonzero(mat_slice[t_blk, :])[0]
                    mat_esell[idx_blk * blk_h + t_blk, idx_cw * chk_w:idx_cw * chk_w + nnz] = mat_slice[t_blk, np.nonzero(mat_slice[t_blk, :])]

            permut_seq = np.argsort(-vec_nnz)
            mat_col_idx_permut[idx_blk * blk_h:(idx_blk + 1) * blk_h, idx_cw * chk_w:(idx_cw + 1) * chk_w] = mat_col_idx[idx_blk * blk_h + permut_seq, idx_cw * chk_w:(idx_cw + 1) * chk_w]
            mat_esell[idx_blk * blk_h:(idx_blk + 1) * blk_h, idx_cw * chk_w:(idx_cw + 1) * chk_w] = mat_esell[idx_blk * blk_h + permut_seq, idx_cw * chk_w:(idx_cw + 1) * chk_w]
            mat_permut[idx_blk * blk_h:(idx_blk + 1) * blk_h, idx_cw] = permut_seq

            # fill in the nnz matrix
            mat_nnz_permut[idx_blk * blk_h:(idx_blk + 1) * blk_h, idx_cw] = vec_nnz[permut_seq]
            for t_blk in range(0, blk_h):
                nnz = vec_nnz[permut_seq[t_blk]]
                if nnz == 0:
                    mat_col_idx_encode[idx_blk * blk_h + t_blk, idx_cw] = idx_encodebook[nnz]
                else:
                    col_idx_vld = tuple(mat_col_idx_permut[idx_blk * blk_h + t_blk, idx_cw * chk_w: idx_cw * chk_w + nnz])
                    mat_col_idx_encode[idx_blk * blk_h + t_blk, idx_cw] = idx_encodebook[nnz][col_idx_vld]

            # slicing the Chunk, be excuted in each BLK (idx_blk)
            for idx_nchk in range(0, n_chk):
                # STEP3: obtain CHK_w
                chk_max_width = mat_nnz_permut[idx_blk * blk_h + idx_nchk * chk_h, idx_cw]
                mat_chkw[idx_blk * n_chk + idx_nchk, idx_cw] = chk_max_width
                # record the overhead
                vec_nnz_permut = vec_nnz[permut_seq]
                for t_chk in range(0, chk_h):
                    mat_chk_overhead[idx_blk * blk_h + idx_nchk * chk_h + t_chk, idx_cw] = chk_max_width - vec_nnz_permut[idx_nchk * chk_h + t_chk]

    # print("Finish eSELL format construction.")
    return mat_esell, mat_permut, mat_chkw, mat_col_idx_encode, mat_chk_overhead


def esell_computation(mat_esell, mat_permut, mat_chkw, mat_col_idx_encode, chk_h, n_chk, chk_w, idx_decodebook, vec_x):
    '''
    :param mat_esell:
    :param mat_permut:
    :param mat_chkw:
    :param mat_col_idx_encode:
    :param chk_h:
    :param n_chk:
    :param chk_w:
    :param idx_decodebook:
    :param vec_x:
    :return:
    '''

    # TODO: change the mat_esell to a linear input
    (row, col) = mat_esell.shape
    blk_h = n_chk * chk_h

    # result vector
    vec_y = np.zeros(row, dtype='int32')

    for idx_cw in range(0, int(col / chk_w)):
        for idx_blk in range(0, int(row / blk_h)):
            # obtain the permutation table
            permut_seq = mat_permut[idx_blk * blk_h:(idx_blk + 1) * blk_h, idx_cw]
            for idx_nchk in range(0, n_chk):

                for jj_chkh in range(0, chk_h):
                    # obtain the col_idx_code & decode
                    col_idx_code = mat_col_idx_encode[idx_blk * blk_h + idx_nchk * chk_h + jj_chkh, idx_cw]
                    col_idx = idx_decodebook[:, col_idx_code]
                    # number of iteration of ii_chkw depends on the records in mat_chkw
                    for ii_chkw in range(0, mat_chkw[idx_blk * n_chk + idx_nchk, idx_cw]):
                        elem_x = vec_x[idx_cw * chk_w + col_idx[ii_chkw]]
                        multi_res = mat_esell[idx_blk * blk_h + idx_nchk * chk_h + jj_chkh, idx_cw * chk_w + ii_chkw] * int(elem_x)
                        vec_y[idx_blk * blk_h + permut_seq[idx_nchk * chk_h + jj_chkh]] = vec_y[idx_blk * blk_h + permut_seq[idx_nchk * chk_h + jj_chkh]] + multi_res

    # print("Finish computation with eSELL format.")
    return vec_y


if __name__ == '__main__':
    # variable from input arguments
    argv = sys.argv
    ROW = int(argv[1])
    COL = int(argv[2])
    DENSITY = float(argv[3])

    # define the idx_encodebook (i.e. chk_w=4)
    idx_encodebook = []
    idx_encodebook.append(0)

    tmp_mat = np.zeros([4], dtype='int16')
    tmp_mat[:] = [0, 4, 6, 7]
    idx_encodebook.append(tmp_mat)

    tmp_mat = np.zeros([4, 4], dtype='int16')
    tmp_mat[0, 1] = 0
    tmp_mat[0, 2] = 2
    tmp_mat[0, 3] = 3
    tmp_mat[1, 2] = 4
    tmp_mat[1, 3] = 5
    tmp_mat[2, 3] = 6
    idx_encodebook.append(tmp_mat)

    tmp_mat = np.zeros([4, 4, 4], dtype='int16')
    tmp_mat[0, 1, 2] = 0
    tmp_mat[0, 1, 3] = 1
    tmp_mat[0, 2, 3] = 2
    tmp_mat[1, 2, 3] = 4
    idx_encodebook.append(tmp_mat)

    tmp_mat = np.zeros([4, 4, 4, 4], dtype='int16')
    tmp_mat[0, 1, 2, 3] = 0
    idx_encodebook.append(tmp_mat)

    # define the idx_decodebook (chk_w=4)
    idx_decodebook = np.zeros([4, 8], dtype='int16')
    idx_decodebook[0][[4, 5]] = 1
    idx_decodebook[0][6] = 2
    idx_decodebook[0][7] = 3
    idx_decodebook[1][[0, 1]] = 1
    idx_decodebook[1][[2, 4]] = 2
    idx_decodebook[1][[3, 5, 6]] = 3
    idx_decodebook[2][0] = 2
    idx_decodebook[2][[1, 2, 4]] = 3
    idx_decodebook[3][[0]] = 3

    for idx_hw_ratio in [1]:
        for idx_rowcol in [1]:
            for DENSITY in [1]:

                ROW = 100*4; COL = 100; DENSITY = 0.4

                # ROW = idx_rowcol * 128 * idx_hw_ratio; COL = idx_rowcol * 128

                # generate the sparse matrix in COO format
                mat_coo = sparse.rand(ROW, COL, density=DENSITY, format='coo', random_state=None)

                # trans float to fixed point, sparse.rand only support float random
                mat_coo.data = frac2fix(mat_coo.data, 'int16')
                mat_csr = mat_coo.tocsr()
                mat_csc = mat_coo.tocsc()
                mat_array = mat_coo.toarray()

                # generate dense vector
                vec_float = np.random.rand(COL)
                vec_int = frac2fix(vec_float, 'int16')
                # SpMV result
                res = np.dot(mat_array.astype(np.dtype('int32')), vec_int.astype(np.dtype('int32')))

                (chk_h, n_chk, chk_w) = (4, 4, 4)

                # trans to eSELL-C-\sigma format
                mat_esell, mat_permut, mat_chkw, mat_col_idx_encode, mat_chk_overhead = esell_construct(mat_array, idx_encodebook, chk_h, n_chk, chk_w)

                # # verification
                # vec_y = esell_computation(mat_esell, mat_permut, mat_chkw, mat_col_idx_encode, chk_h, n_chk, chk_w, idx_decodebook, vec_int)
                # diff = vec_y - res
                # assert (diff.all() == 0)
                # print("Success.")

                # obtain the storage cost

                # eSELL
                bit_value = 16
                bit_permut = int(math.ceil(np.log2(chk_h * n_chk)))
                bit_encode = 3
                bit_idx = int(math.ceil(np.log2(chk_w)))

                bit_ptr = int(math.ceil(np.log2(np.sum(mat_chkw)) * chk_h * bit_value / 8))

                vol_value = np.sum(mat_chkw) * chk_h * bit_value
                vol_overhead = np.sum(mat_chk_overhead) * bit_value
                vol_permut = mat_permut.size * bit_permut
                vol_col_idx_encode = mat_col_idx_encode.size * bit_encode
                # in general SELL
                vol_col_idx = np.sum(mat_chkw) * chk_h * bit_idx
                # ptr
                vol_ptr = int(COL/chk_w) * bit_ptr
                total_eSELL = vol_value + vol_permut + vol_col_idx_encode + vol_ptr

                # CSC
                bit_csc_idx = int(math.ceil(np.log2(ROW)))

                vol_csc_value = mat_csc.nnz * bit_value
                vol_csc_idx = mat_csc.indices.size * bit_csc_idx
                vol_csc_ptr = mat_csc.indptr.size * bit_ptr
                total_CSC = vol_csc_value + vol_csc_idx + vol_csc_ptr

                # Hansong's format
                bit_han_idx = 4
                vol_padding = 0
                for ii in range(0, mat_csc.indices.size-1):
                    if mat_csc.indices[ii+1] - mat_csc.indices[ii] > 2 ** bit_han_idx:
                        vol_padding = vol_padding + 1

                vol_han_value = (mat_csc.nnz + vol_padding) * bit_value
                vol_han_overhead = vol_padding * bit_value
                vol_han_idx = (mat_csc.nnz + vol_padding) * bit_han_idx
                vol_han_ptr = mat_csc.indptr.size * bit_ptr
                total_han = vol_han_value + vol_han_idx + vol_han_ptr

                # # Blocked-CSC (deprecated)
                # row_blk_bcsc = chk_w
                #
                # vol_bcsc_idx = mat_csc.indices.size * int(np.log2(row_blk_bcsc))
                # vol_bcsc_ptr = mat_csc.indptr.size * int(np.log2(ROW*COL/row_blk_bcsc)) * int(ROW/row_blk_bcsc)

                # dense
                vol_dense_value = bit_value * ROW * COL
                total_dense = vol_dense_value

                # calculate the word (64bit/word in risc-v)
                # eSELL
                # head info len
                if (chk_w == 4) & (n_chk <= 4):
                    head_info_len_eSELL = int(mat_chkw.size / 2)
                else:
                    head_info_len_eSELL = mat_chkw.size
                len_eSELL = head_info_len_eSELL + int(vol_value / 64)

                # CSC
                # head info len
                head_info_len_csc = math.ceil(mat_csc.indices.size / int(64 / bit_csc_idx))
                len_CSC = head_info_len_csc + int(vol_csc_value / 64)

                # Hansong's
                # head info len
                val_per_word = int(64 / (bit_value + bit_han_idx))
                len_han = int((mat_csc.nnz + vol_padding) / val_per_word)

                # dense
                len_dense = int(vol_dense_value / 64)

                print(len_eSELL)

                # print(ROW, COL, DENSITY, vol_dense_value, len_dense, vol_value, vol_overhead, vol_permut, vol_col_idx_encode, vol_col_idx, vol_ptr, len_eSELL, vol_han_value, vol_han_overhead, vol_han_idx, vol_han_ptr, len_han, vol_csc_value, vol_csc_idx, vol_csc_ptr, len_CSC, total_dense, total_eSELL, total_han, total_CSC)
    # print("Script finish.")
