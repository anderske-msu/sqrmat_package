import numpy as np
import PyTPSA
import sys


def uni_transform(ind_i, ind_j, ratio_left, ratio_right, mat, leftm, rightm):
    rightm[:, ind_j] += rightm[:, ind_i] * (ratio_right)

    leftm[ind_i, :] += leftm[ind_j, :] * ratio_left

    temp1 = mat[ind_j, :] * ratio_left  # left
    temp2 = mat[:, ind_i] * (ratio_right)  # right
    temp3 = mat[ind_j, ind_i] * ratio_left * ratio_right

    mat[ind_i, :] += temp1  # update left
    mat[:, ind_j] += temp2
    mat[ind_i, ind_j] += temp3


def jordan_norm_form(sqrmat_ori, epsilon=0.0):
    sqrmat = sqrmat_ori * 1.0

    dim, dimy = sqrmat.shape

    if dim != dimy:
        print("Not a square matrix")
        return

    # if dim==1:
    #    return np.eye(1, dtype=complex), sqrmat, np.eye(1, dtype=complex)

    jnf_leftm = np.eye(dim, dtype=complex)
    jnf_rightm = np.eye(dim, dtype=complex)

    i = 1
    siup = sqrmat[i - 1, i]
    if np.abs(siup) > epsilon:
        uni_transform(
            i - 1, i - 1, 1 / siup - 1, siup - 1, sqrmat, jnf_leftm, jnf_rightm
        )
        # sqrmat[i-1,i]=sympy.simplify(sqrmat[i-1,i].expand(complex=complex_expand))

    for i in range(2, dim):
        # How many non zero terms in ith column row 0 to i-2
        non_zero_list = []
        js = i
        for j in range(i):
            if np.abs(sqrmat[j, i]) > epsilon:
                # if j in previous_zero_terms:

                if j < i - 1 and np.abs(sqrmat[j, j + 1] - 1) < 1e-10:
                    # print(j,i)
                    old_i = j + 1
                    ratio = sqrmat[j, i]
                    uni_transform(
                        old_i, i, ratio, -ratio, sqrmat, jnf_leftm, jnf_rightm
                    )
                else:
                    non_zero_list.append(j)
                    # print(j, i, non_zero_list)

        if len(non_zero_list) == 0:
            continue
        elif len(non_zero_list) == 1:
            j = non_zero_list[0]
            ratio = sqrmat[j, i]
            uni_transform(i, i, ratio - 1, 1 / ratio - 1, sqrmat, jnf_leftm, jnf_rightm)
            js = j

        else:

            def count_chain_length(k, eps):
                chain_length = 1
                for kk in range(k, 0, -1):
                    if np.abs(sqrmat[kk - 1, kk]) <= eps:
                        break
                    else:
                        chain_length += 1
                return chain_length

            # print("Need special treatment")

            ind = 1
            je = non_zero_list[0]  # To be eliminated
            js = -1

            while ind < len(non_zero_list):
                js = non_zero_list[ind]  # To stay

                le = count_chain_length(je, epsilon)
                ls = count_chain_length(js, epsilon)
                # print(je, js, le, ls)
                if le > ls:
                    le, ls = ls, le
                    je, js = js, je

                # print(je, js, le, ls)
                ratio = -sqrmat[je, i] / sqrmat[js, i]

                for k in range(le):
                    uni_transform(
                        je - k, js - k, ratio, -ratio, sqrmat, jnf_leftm, jnf_rightm
                    )

                je = js
                ind += 1

            ratio = sqrmat[js, i]
            uni_transform(i, i, ratio - 1, 1 / ratio - 1, sqrmat, jnf_leftm, jnf_rightm)

        if js < i - 1:
            # print(js,i,"special")
            leftm = np.eye(dim, dtype=complex)

            temp = leftm[i, :] * 1
            leftm[js + 2 : i + 1, :] = leftm[js + 1 : i, :] * 1
            leftm[js + 1, :] = temp * 1
            rightm = leftm.transpose()
            jnf_rightm = jnf_rightm.dot(rightm)
            jnf_leftm = leftm.dot(jnf_leftm)
            sqrmat = leftm.dot(sqrmat).dot(rightm)

    return jnf_leftm, sqrmat, jnf_rightm


def jordan_chain_structure(jnf, epsilon):
    dim, _ = jnf.shape
    structure = []
    new_chain = [
        0,
    ]
    for i in range(1, dim):
        if np.abs(jnf[i - 1, i]) <= epsilon:
            structure.append(new_chain)
            new_chain = [
                i,
            ]
        else:
            new_chain.append(i)
    structure.append(new_chain)
    return structure


class tpsvar(object):
    epsilon = 1e-14

    def __init__(self, dim, order):
        PyTPSA.initialize(2 * dim, order)
        self.dim = dim
        self.order = order
        self.vars = [PyTPSA.tpsa(0.0, i + 1, dtype=complex) for i in range(2 * dim)]

        self.sqrmat_dim = PyTPSA.tpsa.get_max_terms()

    def get_degenerate_list(self, target, resonance=None):
        self.target = target
        if target > self.dim or target <= 0:
            print(f"Wrong target specified, should be one of the {self.dim} dimensions")
            return
        powerlist = np.array(
            [PyTPSA.tpsa.get_power_index(i) for i in range(self.sqrmat_dim)]
        )
        fns = np.zeros((self.sqrmat_dim, self.dim), dtype=int)
        for i in range(self.dim):
            fns[:, i] = powerlist[:, i * 2 + 1] - powerlist[:, i * 2 + 2]
        fns[:, target - 1] -= 1
        # fnx = powerlist[:, 1] - powerlist[:, 2]
        # fny = powerlist[:, 3] - powerlist[:, 4]

        if resonance is not None:
            res = np.array(resonance, dtype=int)
            if len(resonance) != self.dim:
                print(
                    f"Resonance dimension {len(resonance)} is not compatible with the variable dimension {self.dim}"
                )
            if np.any(res) == False:
                pass
            else:
                absres2 = res.dot(res)
                absfn2 = (fns * fns).sum(axis=1)
                fndotres = fns.dot(res)
                sign = np.sign(fndotres)
                mask1 = absfn2 > 0
                mask2 = fndotres * fndotres == absres2 * absfn2
                mask = np.logical_and(mask1, mask2)

                ratio = sign[mask] * (
                    np.sqrt(np.floor_divide(absfn2[mask], absres2))
                ).astype(int)
                fns[mask] -= np.outer(ratio, res)

        mask = 1 - np.any(fns, axis=1)
        self.degenerate_list = np.nonzero(mask)[0]

    def get_variables(self, Uinv=None):
        if Uinv is None:
            _vars = []
            for i in range(self.dim):
                _vars.append((self.vars[2 * i] + self.vars[2 * i + 1]) / 2.0)
                _vars.append((self.vars[2 * i] - self.vars[2 * i + 1]) / 2.0 * 1j)

        else:
            _vars = Uinv.dot([self.vars[i] for i in range(2 * self.dim)]).tolist()

        return _vars

    def construct_sqr_matrix(self, periodic_map):
        if len(periodic_map) != 2 * self.dim:
            print(
                f"Incorrect length of map, should be same as the dimension {self.dim}"
            )
        sqrmat = np.zeros((self.sqrmat_dim, self.sqrmat_dim), dtype=complex)
        for i in range(self.sqrmat_dim):
            pind = PyTPSA.tpsa.get_power_index(i)
            newvar = PyTPSA.tpsa(1.0, dtype=complex)
            for j in range(len(periodic_map)):
                for k in range(pind[j + 1]):
                    newvar *= periodic_map[j]
            indices = newvar.indices
            sqrmat[i, 0 : len(indices)] = indices
        self.sqr_mat = sqrmat

        return

    def sqrmat_reduction(self):
        self.sqr_mat_omegaI = self.sqr_mat - self.sqr_mat[
            self.target * 2 - 1, self.target * 2 - 1
        ] * np.eye(self.sqrmat_dim, dtype=complex)
        self.right_mat = np.eye(self.sqrmat_dim, dtype=complex)
        self.left_mat = np.eye(self.sqrmat_dim, dtype=complex)

        for ind_i in reversed(self.degenerate_list):
            for ind_j in range(ind_i + 1, self.sqrmat_dim):
                if np.abs(self.sqr_mat_omegaI[ind_i, ind_j]) <= tpsvar.epsilon:
                    self.sqr_mat_omegaI[ind_i, ind_j] *= 0.0
                    continue
                if ind_j in self.degenerate_list:
                    continue

                ratio = (
                    -self.sqr_mat_omegaI[ind_i, ind_j]
                    / self.sqr_mat_omegaI[ind_j, ind_j]
                )
                uni_transform(
                    ind_i,
                    ind_j,
                    ratio,
                    -ratio,
                    self.sqr_mat_omegaI,
                    self.left_mat,
                    self.right_mat,
                )

        for ind_j in self.degenerate_list:
            for ind_i in range(ind_j - 1, -1, -1):
                if np.abs(self.sqr_mat_omegaI[ind_i, ind_j]) <= tpsvar.epsilon:
                    self.sqr_mat_omegaI[ind_i, ind_j] *= 0.0
                    continue
                if ind_i in self.degenerate_list:
                    continue

                ratio = (
                    self.sqr_mat_omegaI[ind_i, ind_j]
                    / self.sqr_mat_omegaI[ind_i, ind_i]
                )
                uni_transform(
                    ind_i,
                    ind_j,
                    ratio,
                    -ratio,
                    self.sqr_mat_omegaI,
                    self.left_mat,
                    self.right_mat,
                )

        self.reduced_mat = self.sqr_mat_omegaI[self.degenerate_list, :][
            :, self.degenerate_list
        ]
        if tpsvar.epsilon > 0:
            self.reduced_mat = np.around(
                self.reduced_mat, decimals=int(-np.log10(tpsvar.epsilon))
            )
            self.left_mat = np.around(
                self.left_mat, decimals=int(-np.log10(tpsvar.epsilon))
            )
            self.right_mat = np.around(
                self.right_mat, decimals=int(-np.log10(tpsvar.epsilon))
            )
        return

    def jordan_form(self):
        if len(self.degenerate_list) == 1:
            # No need for to find jordan form:
            self.right_vector = self.right_mat[:, self.degenerate_list[0]] * 1
            self.left_vector = self.left_mat[self.degenerate_list[0], :] * 1
            return

        self.jnf_leftm, self.jnf_mat, self.jnf_rightm = jordan_norm_form(
            self.reduced_mat, epsilon=tpsvar.epsilon
        )
        if tpsvar.epsilon > 0:
            self.jnf_leftm = np.around(
                self.jnf_leftm, decimals=int(-np.log10(tpsvar.epsilon))
            )
            self.jnf_rightm = np.around(
                self.jnf_rightm, decimals=int(-np.log10(tpsvar.epsilon))
            )
            self.jnf_mat = np.around(
                self.jnf_mat, decimals=int(-np.log10(tpsvar.epsilon))
            )

        # temp_mat = sympy.Matrix(len(self.terms), len(self.degenerate_list), lambda i, j: 0)
        temp_mat = np.zeros((self.sqrmat_dim, len(self.degenerate_list)), dtype=complex)
        # return type(temp_mat[self.zero_term_index,:])
        # temp_mat[self.zero_term_index,:]=self.jnf_rightm[:,:]

        for i in range(len(self.degenerate_list)):
            temp_mat[self.degenerate_list[i], :] = self.jnf_rightm[i, :]

        self.right_vector = self.right_mat.dot(temp_mat)

        # temp_mat = sympy.Matrix(len(self.degenerate_list), len(self.terms), lambda i, j: 0)
        temp_mat = np.zeros((len(self.degenerate_list), self.sqrmat_dim), dtype=complex)
        for i in range(len(self.degenerate_list)):
            temp_mat[:, self.degenerate_list[i]] = self.jnf_leftm[:, i]
        # temp_mat[:, self.zero_term_index] = self.jnf_leftm
        self.left_vector = temp_mat.dot(self.left_mat)
        return
