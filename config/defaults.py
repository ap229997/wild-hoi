# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
# from yacs.config import CfgNode as CN
from fvcore.common.config import CfgNode as CN

_C = CN()

_C.GPU = 0
_C.EXP = 'tmp'
_C.DUM = ''
_C.MODEL_SIG = ''
_C.MODEL_PATH = ''
_C.OUTPUT_DIR = 'debug'

_C.SEED = 123

_C.OG_RUN = False # whether to run original code
_C.ADV = False
_C.CONTACT_ADV = False
_C.XSEC = False
_C.XSEC_VOLUME = False
_C.MULTIVIEW = False
_C.NVIEWS = False
_C.OBJ_GEN = False
_C.USE_GT_CAM = False
_C.SMOOTHEN_OCC = False
_C.RETURN_OCC = False
_C.FACTORED = False
_C.FACTORED_CAT = False
_C.BATCH_IMG = False
_C.IMG_CAT = False

_C.TRAIN = CN()
_C.TRAIN.PRINT_EVERY = 100
_C.TRAIN.EVAL_EVERY = 10
_C.TRAIN.VIS_EVERY = 100
_C.TRAIN.ITERS = 50000
_C.TRAIN.EPOCH = 200


_C.LOSS = CN()
_C.LOSS.OFFSCREEN = 'gt'  # [gt /out / idc]
_C.LOSS.KL = 1e-4
_C.LOSS.RECON = 0.0
_C.LOSS.INTER = None
_C.LOSS.SEG = 0.0
_C.LOSS.ENFORCE_MINMAX = True
_C.LOSS.SDF_MINMAX = 0.1
_C.LOSS.OCC = 'strict' # don't know what this is used for
_C.LOSS.ENC = 0.0
_C.LOSS.DISC = 0.0
_C.LOSS.EIK = 0.0
_C.LOSS.MULTIVIEW = 0.0
_C.LOSS.NVIEWS = 0.0
_C.LOSS.REG = 0.0
_C.LOSS.OCCUPANCY = 0.0
_C.LOSS.OCC_CONSISTENCY = 0.0
_C.LOSS.RAY_CONSISTENCY = 0.0
_C.LOSS.N_RAY = 0
_C.LOSS.N_RAY_PTS = 0

# optimization loss
_C.OPT = CN()
_C.OPT.NAME = 'opt'
_C.OPT.STEP = 1000
_C.OPT.LR = 1e-3
_C.OPT.NET = False
_C.OPT.BATCH_SIZE = 16
_C.OPT.INIT = 'zero'

# deprecate
_C.OPT.OPT = 'adam'


_C.DB = CN()
_C.DB.CLS = ''
_C.DB.NAME = 'obman'
_C.DB.TESTNAME = 'obman'
_C.DB.DIR = ''  # change to your path
_C.DB.RADIUS = 0.2
_C.DB.CACHE = True
_C.DB.IMAGE = False
_C.DB.INPUT = 'rgb'  # rgb, rgba, flow
_C.DB.CUSTOM = None
_C.DB.SAMPLER = None
_C.DB.NO_HANDS = False
_C.DB.PERCENT = 0.0

_C.DB.NUM_POINTS = 12000

# refine
_C.DB.JIT_ART = 0.1  # simulate prediction error
_C.DB.JIT_P = 0  # simulate prediction error
_C.DB.JIT_SCALE = 0.5  # simulate prediction error
_C.DB.JIT_TRANS = 0.2  # simulate prediction error


_C.MODEL = CN()
_C.MODEL.NAME = 'IHoi'
_C.MODEL.DEC = 'PixCoord'
_C.MODEL.ENC = 'ImageSpEnc'
_C.MODEL.ENC_RESO = -3
_C.MODEL.OCCUPANCY = False

_C.MODEL.FRAME = 'norm'  # norm / hand / obj
_C.MODEL.BATCH_SIZE = 64
_C.MODEL.Z_DIM = 256
_C.MODEL.THETA_DIM = 45
_C.MODEL.THETA_EMB = 'pca'
_C.MODEL.PC_DIM = 128
_C.MODEL.LATENT_DIM = 128
_C.MODEL.FREQ = 10
_C.MODEL.IS_PCA = 0
_C.MODEL.GRAD = 'none'
_C.MODEL.SUBSAMPLE = None

_C.MODEL.SDF = CN()
_C.MODEL.SDF.DIMS = (512, 512, 512, 512, 512, 512, 512, 512, )
_C.MODEL.SDF.SKIP_IN = (4, )
_C.MODEL.SDF.GEOMETRIC_INIT = False
_C.MODEL.SDF.th = True


_C.CAMERA = CN()
_C.CAMERA.F = 100.

_C.RENDER = CN()
_C.RENDER.METRIC = 1  # CM, 1000-MM.



_C.HAND = CN()
_C.HAND.WRAP = 'mano'
_C.HAND.MANO_PATH = '../data/smplx/mano'


_C.SOLVER = CN()
_C.SOLVER.BASE_LR = 1e-4
_C.SOLVER.ENC_LR = 1e-4
_C.SOLVER.DISC_LR = 1e-4
_C.SOLVER.NUM_WORKERS = 0
_C.SOLVER.DEBUG = False
_C.SOLVER.DISC_ITER = 1
_C.SOLVER.ENC_ITER = 1


# test specific parameters
_C.TEST = CN()
_C.TEST.NAME = 'default'
_C.TEST.DIR = ''
_C.TEST.SET = 'test'
_C.TEST.NUM = 2



def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
