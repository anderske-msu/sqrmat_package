import numpy as np
from numpy.typing import ArrayLike
import sympy as sp  # For test_u
import time  # For timing functions
import PyTPSA
from .sqrmat import jordan_chain_structure, tpsvar


def numpify(tps):
    """
    Yue's function for turning TPS function into numpy function
    """
    mask = np.abs(tps.indices) > 1e-16
    coeffs = np.array(tps.indices)[mask].tolist()
    inds = np.arange(len(tps.indices))[mask]
    powers = [tps.find_power(i)[1:] for i in inds]
    dim = int(tps.get_dim())

    def pyfunc(list_of_num):
        result = 0 + 0j
        for i in range(len(coeffs)):
            temp = coeffs[i]
            for j in range(dim):
                temp *= list_of_num[j] ** powers[i][j]
            result += temp
        return result

    return pyfunc


class square_matrix:
    def __init__(self, dim: int, order: int):
        self.dim = dim  # Number of spacial dimentions
        self.order = order  # Square Matrix order

        self._hp = tpsvar(dim, order=order)
        self.variables = self._hp.vars

        self.degenerate_list = [None for _ in range(dim)]
        self.left_vector = [None for _ in range(dim)]
        self.left_vector_second = [None for _ in range(dim)]

        self.jordan_norm_form_matrix = [None for _ in range(dim)]
        self.jordan_chain_structure = [None for _ in range(dim)]

        self.__weights = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.complex128)

        self.square_matrix = None

        self.__fztow = [None for _ in range(dim)]
        self.__wftoz = [None for _ in range(dim)]
        self.__fzmap = [None for _ in range(dim)]

    def construct_square_matrix(self, periodic_map: list):
        self.__fzmap = [numpify(f) for f in periodic_map]
        self._hp.construct_sqr_matrix(periodic_map=periodic_map)
        self.square_matrix = self._hp.sqr_mat

    def get_transformation(
        self, res=None, left_eigenvector: int = 0, epsilon: float = 1e-15
    ):
        """
        Creates the transformation for the approximate action.

        Inputs:
            res: list
                The resonance the linear tunes are on. Default is None

            left_eigenvector: int
                The left eigenvector (row of U) in the first jordan chain to be used. Default is the first (0).

            epsilon: float
                Epsilon for the finding the jordan chain structure. Default is 1e-15.

        Outputs:
            None
        """

        # TODO Allow the use of mulitple Jordan Chains in the transformation

        # Check if contruct sqrmat was run
        if self.square_matrix is None:
            err = 'The square matrix has not been constructed. Run "construct_square_matrix" first.'
            raise Exception(err)

        for di in range(self.dim):
            self._hp.get_degenerate_list(di + 1, resonance=res)
            self._hp.sqrmat_reduction()
            self._hp.jordan_form()

            # ? Are these Better as lists or np.ndarrays?
            self.degenerate_list[di] = self._hp.degenerate_list.copy()
            self.jordan_norm_form_matrix[di] = self._hp.jnf_mat.copy()
            self.left_vector[di] = self._hp.left_vector[left_eigenvector].copy()
            self.left_vector_second[di] = self._hp.left_vector[
                left_eigenvector + 1
            ].copy()  # TODO make left_eig a list to choose (not plus 1)
            self.jordan_chain_structure[di] = jordan_chain_structure(
                self.jordan_norm_form_matrix[di], epsilon=epsilon
            )

        # TODO Rewrite to make sure it'll work with dim != 2

        # First left eigenvector
        # ? Should I use complex or np.complex128?
        wx0z = PyTPSA.tpsa(input_map=self.left_vector[0], dtype=complex)
        wx0cz = wx0z.conjugate(mode="CP")

        wy0z = PyTPSA.tpsa(input_map=self.left_vector[1], dtype=complex)
        wy0cz = wy0z.conjugate(mode="CP")

        # Second left eigenvector
        wx1z = PyTPSA.tpsa(input_map=self.left_vector_second[0], dtype=complex)
        wx1cz = wx0z.conjugate(mode="CP")

        wy1z = PyTPSA.tpsa(input_map=self.left_vector_second[1], dtype=complex)
        wy1cz = wy0z.conjugate(mode="CP")

        self.__w0list = [wx0z, wx0cz, wy0z, wy0cz]
        self.__w1list = [wx1z, wx1cz, wy1z, wy1cz]

        invw0z = PyTPSA.inverse_map(self.__w0list)

        self.__fztow0 = [numpify(f) for f in self.__w0list]
        self.__fztow1 = [numpify(f) for f in self.__w1list]

        self.__fztow = 1 * self.__fztow0
        self.__wftoz = [numpify(f) for f in invw0z]

        # Build Jacobian function
        # ! Used for netwons inverse, needs to be update for using weights
        self.__fjacobian = [
            numpify(wtemp.derivative(i + 1))
            for wtemp in self.__w0list
            for i in range(2 * self.dim)
        ]

    def w(self, z: ArrayLike, weights: np.ndarray = None) -> np.ndarray:
        """Transforms normalized complex coordinates into the transformed phase space.

        Inputs:
            z: array_like
                [zx, zx*, zy, zy*]

            weights: numpy.ndarray
                The weights for each left vector transformation to be used in the full w transformation. Dimentions are (dim, 4).
                The default is wj = wj0 where j is each of the spatial dimentions.
                    i.e. wj(z) = weights[j,0] * wx0(x) + weights[j,1] * wx1(x) + weights[j,2] * wy0(x) + weights[j,3] * wy1(x)

        Returns:
            w: numpy.ndarray
                [wx, wx*, wy, wy*]
        """

        if self.__fztow[0] is None:
            err = (
                'The transformation has not been found. Run "get_transformation" first.'
            )
            raise Exception(err)

        if (weights is not None) and np.any(self.__weights != weights):
            start = time.perf_counter()

            print("Transformation weights are being updated.")

            # Checks if the weights have changed
            self.__weights = np.copy(weights)

            # Finding new transformation and inverse functions

            wz = []  # PyTPSA w terms

            for di in range(self.dim):
                wz.append(
                    self.__weights[di, 0] * self.__w0list[0]
                    + self.__weights[di, 1] * self.__w1list[0]
                    + self.__weights[di, 2] * self.__w0list[2]
                    + self.__weights[di, 3] * self.__w1list[2]
                )

            wcz = [wi.conjugate(mode="CP") for wi in wz]  # Conjugate terms

            wlist = []  # New w list

            for wi, wci in zip(wz, wcz):
                # Adds new W PyTPSA terms alternating between w and w*
                wlist.append(wi)
                wlist.append(wci)

            invz = PyTPSA.inverse_map(wlist)  # New inverse function

            # Update ztow functions and inverse
            self.__fztow = [numpify(f) for f in wlist]
            self.__wftoz = [numpify(f) for f in invz]

            end = time.perf_counter()

            print(f"Weights Updated: Time {end-start}s")

        return np.array(
            [
                self.__fztow[0](z),
                self.__fztow[1](z),
                self.__fztow[2](z),
                self.__fztow[3](z),
            ]
        )

    def z(self, w: ArrayLike) -> np.ndarray:
        """Tranformes transformed coordinates into the normalized complex coordinate phase space.
        Assumes the weights are the same ones used in the last call of self.w

        Inputs:
            w: array_like
                [wx, wx*, wy, wy*]

        Returns:
            z: numpy.ndarray
                [zx, zx*, zy, zy*]
        """

        if self.__wftoz[0] is None:
            err = (
                'The transformation has not been found. Run "get_transformation" first.'
            )
            raise Exception(err)

        return np.array(
            [
                self.__wftoz[0](w),
                self.__wftoz[1](w),
                self.__wftoz[2](w),
                self.__wftoz[3](w),
            ]
        )

    def get_weights(
        self, z: ArrayLike, nalpha: int, nbeta: int, a: np.ndarray
    ) -> np.ndarray:
        # ! Not working
        start = time.perf_counter()

        num_eig = 4  # Number of left eigenvectors being used in the linear combination of the transformation

        wx0 = np.array(self.__fztow0[0](z))
        wx1 = np.array(self.__fztow1[0](z))
        wy0 = np.array(self.__fztow0[2](z))
        wy1 = np.array(self.__fztow1[2](z))

        fwx0 = np.reshape(wx0, (nalpha, nbeta))
        fwx1 = np.reshape(wx1, (nalpha, nbeta))
        fwy0 = np.reshape(wy0, (nalpha, nbeta))
        fwy1 = np.reshape(wy1, (nalpha, nbeta))

        fwx0 = np.fft.fft2(fwx0)
        fwx1 = np.fft.fft2(fwx1)
        fwy0 = np.fft.fft2(fwy0)
        fwy1 = np.fft.fft2(fwy1)

        wf = np.array([fwx0, fwx1, fwy0, fwy1])

        # Build F1/2 Matricies
        F1 = np.zeros((num_eig, num_eig), dtype=np.complex128)
        F2 = np.zeros_like(F1)

        for j in range(num_eig):
            for h in range(num_eig):
                sum1 = 0
                sum2 = 0
                for n in range(nalpha):
                    for m in range(nbeta):
                        term = np.conj(wf[j, n, m]) * wf[h, n, m]

                        if np.abs(n - 1) + np.abs(m) != 0:
                            sum1 += term

                        if np.abs(n) + np.abs(m - 1) != 0:
                            sum2 += term

                F1[j, h] = sum1
                F2[j, h] = sum2

        print("det", np.linalg.det(F1), np.linalg.det(F2))
        F1inv = np.linalg.inv(F1)
        F2inv = np.linalg.inv(F2)

        a1 = []
        a2 = []
        for h in range(num_eig):
            a1num = np.sum(F1inv[h, :] * np.conj(wf[:, 1, 0]))
            a2num = np.sum(F2inv[h, :] * np.conj(wf[:, 0, 1]))

            a1den = 0
            a2den = 0
            for m in range(num_eig):
                for j in range(num_eig):
                    a1den += wf[m, 1, 0] * F1inv[m, j] * np.conj(wf[j, 1, 0])
                    a2den += wf[m, 0, 1] * F2inv[m, j] * np.conj(wf[j, 0, 1])

            a1.append(a1num / a1den)
            a2.append(a2num / a2den)

            # print(f'a1{h}= {a1num}/{a1den}')
            # print(f'a2{h}= {a2num}/{a2den}')

        na = np.array([a1, a2])

        # Normalization

        # Normalize Vectors
        # v = []
        # nv = []

        # for i in range(self.dim):

        #     v.append(a[i,0]*wx0 + a[i,1]*wx1 + a[i,2]*wy0 + a[i,3]*wy1)
        #     nv.append(na[i,0]*wx0 + na[i,1]*wx1 + na[i,2]*wy0 + na[i,3]*wy1)

        # v = np.array(v)
        # nv = np.array(nv)

        # vnorm = np.linalg.norm(v, axis=0)
        # nvnorm = np.linalg.norm(nv, axis=0)

        # vnorm = np.average(vnorm)
        # nvnorm = np.average(nvnorm)

        # scale_factor = vnorm/nvnorm

        # na *= scale_factor
        # # na /= nvnorm

        # Normalize Weights
        sum_weights = np.sum(na, axis=1)

        for i, sw in enumerate(sum_weights):
            na[i, :] /= sw

        print(f"Here! A1: {na[0,:]}; A2: {na[1,:]}, norm: {sum_weights}")
        # print(f"Here! A1: {na[0,:]}; A2: {na[1,:]}")

        end = time.perf_counter()

        print(f"Weights Found!: Time {end-start}s")

        return na

    def jacobian(self, z: ArrayLike) -> np.ndarray:
        """Returns the value of the jacobian matrix at z.

        Inputs:
            z: array_like; [zx, zx*, zy, zy*]

        Returns:
            jac: numpy.ndarray; Jacobian
        """
        jac = []

        for f in self.__fjacobian:
            jac.append(f(z))

        return np.array(jac)

    def map(self, z: ArrayLike) -> np.ndarray:
        """Runs z through the given one turn map. z' = f(z)

        Inputs:
            z; array_like; [zx, zx*, zy, zy*]

        Returns:
            z'; numpy.ndarray; [zx', zx'*, zy', zy'*]
        """
        if self.__fzmap[0] is None:
            err = (
                'The transformation has not been found. Run "get_transformation" first.'
            )
            raise Exception(err)

        return np.array(
            [
                self.__fzmap[0](z),
                self.__fzmap[1](z),
                self.__fzmap[2](z),
                self.__fzmap[3](z),
            ]
        )

    def test_sqrmat(self, atol: float = 1e-8, results: bool = False) -> tuple:
        """Checks that the lower triagular elements of the square matrix are close to zero.

        Inputs:
            atol: The absolute tolerance to check against.
            results: Print results

        Returns:
            pass: If all the points are close to zero the funtion will return True
        """

        dimx, dimy = self.square_matrix.shape

        pass_ = False

        num_errors = 0
        max_error = 0

        for i in range(dimx):
            for j in range(dimy):
                term = np.abs(self.square_matrix[i, j])

                if i > j:  # On the lower diagonal
                    if not np.isclose(
                        term, 0, atol=atol
                    ):  # If the element is not close to zero count it
                        num_errors += 1

                    if term > max_error:  # Track the largest element
                        max_error = term * 1

        if results:
            print("Number of Errors:", num_errors)
            print("Max Error:", max_error)

        if num_errors == 0:
            pass_ = True

        return pass_, num_errors, max_error

    def test_u(self):
        """Tests U U^(-1) = I"""

        if self.__fzmap[0] is None:
            err = (
                'The transformation has not been found. Run "get_transformation" first.'
            )
            raise Exception(err)

        zx, zy = sp.symbols(r"z_x, z_y")
        zxc = zx.conjugate()
        zyc = zy.conjugate()

        z = (zx, zxc, zy, zyc)

        w = self.w([*z]).tolist()

        ztest = self.z(w).tolist()

        for i in range(self.dim):
            ztest[i] = sp.Poly(ztest[i], z).as_dict()

        return ztest
