import numpy as np

# TODO install package into enviroment
import PyTPSA   # * Must be in the same folder as PyTPSA or in the path

# TODO install these as packages
# ! Import sys and run ' sys.path.insert(1, '../Square-Matrix/Custom_Libraries/') ' and ' sys.path.insert(1, '../Library/CodeLib/') '
from sqrmat import jordan_chain_structure, tpsvar
from commonfuncs import numpify

class square_matrix:

    # TODO Add tracking
    
    def __init__(self, dim, order):
        self.dim = dim
        self.order = order
        
        self._hp = tpsvar(dim, order=order)
        # self.variables = self._hp.get_variables()
        self.variables = self._hp.vars
        
        self.degenerate_list = [None for i in range(dim)]
        self.left_vector = [None for i in range(dim)]

        self.jordan_norm_form_matrix = [None for i in range(dim)]
        self.jordan_chain_structure = [None for i in range(dim)]
        
        self.square_matrix = None

        self.__fztow = [None for i in range(dim)]
        self.__wftoz = [None for i in range(dim)]
        self.__fzmap = [None for i in range(dim)]

    def construct_square_matrix(self, periodic_map):
        self.__fzmap = [numpify(f) for f in periodic_map]
        self._hp.construct_sqr_matrix(periodic_map=periodic_map)
        self.square_matrix = self._hp.sqr_mat

    def get_transformation(self, target=None, res=None, chain=0, dim=None, epsilon=1e-15):

        # Check if contruct sqrmat was run
        if self.square_matrix is None:
            raise Exception("The square matrix has not been constructed. Run \"construct_square_matrix\" first.")

        # First get the degenerate list
        if target is None:
            target = [i for i in range(1, self.dim+1)]
        
        if dim is None:
            dim_iter = self.dim
        else:
            dim_iter = dim

        for di in range(dim_iter):
            self.degenerate_list[di] = self._hp.get_degenerate_list(di+1, resonance=res)
            self._hp.sqrmat_reduction()
            self._hp.jordan_form()

            self.jordan_norm_form_matrix[di] = self._hp.jnf_mat
            self.left_vector[di] = self._hp.left_vector[chain]
            self.jordan_chain_structure[di] = jordan_chain_structure(self.jordan_norm_form_matrix[di], epsilon=epsilon)

        wx0z = PyTPSA.tpsa(input_map=self.left_vector[0], dtype=complex)
        wx0cz = wx0z.conjugate(mode='CP')

        wy0z = PyTPSA.tpsa(input_map=self.left_vector[1], dtype=complex)
        wy0cz = wy0z.conjugate(mode='CP')

        w0list = [wx0z, wx0cz, wy0z, wy0cz]

        invw0z = PyTPSA.inverse_map(w0list)

        self.__fztow = [numpify(f) for f in w0list]
        self.__wftoz = [numpify(f) for f in invw0z]



    def w(self, z):
        if self.__fztow[0] is None:
            raise Exception("The transformation has not been found. Run \"get_transformation\" first.")

        return np.array([self.__fztow[0](z), self.__fztow[1](z), self.__fztow[2](z), self.__fztow[3](z)])

    def z(self, w):
        if self.__wftoz[0] is None:
            raise Exception("The transformation has not been found. Run \"get_transformation\" first.")

        return np.array([self.__wftoz[0](w), self.__wftoz[1](w), self.__wftoz[2](w), self.__wftoz[3](w)])

    def map(self, z):
        if self.__fzmap[0] is None:
            raise Exception("The transformation has not been found. Run \"get_transformation\" first.")

        return np.array([self.__fzmap[0](z), self.__fzmap[1](z), self.__fzmap[2](z), self.__fzmap[3](z)])