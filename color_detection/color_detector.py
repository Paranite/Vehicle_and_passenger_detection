import numpy as np
import numexpr as ne


class ColorDetector:

    @staticmethod
    def bincount_app(a):
        a2D = a.reshape(-1,a.shape[-1])
        col_range = (256, 256, 256) # generically : a2D.max(0)+1
        a1D = np.ravel_multi_index(a2D.T, col_range)
        try:
            res = np.unravel_index(np.bincount(a1D).argmax(), col_range)
        except:
            res = None
        return res

    @staticmethod
    def bincount_numexpr_app(a):
        a2D = a.reshape(-1, a.shape[-1])
        col_range = (256, 256, 256)  # generically : a2D.max(0)+1
        eval_params = {'a0': a2D[:, 0], 'a1': a2D[:, 1], 'a2': a2D[:, 2],
                       's0': col_range[0], 's1': col_range[1]}
        a1D = ne.evaluate('a0*s0*s1+a1*s0+a2', eval_params)
        try:
            return np.unravel_index(np.bincount(a1D).argmax(), col_range)
        except:
            return None