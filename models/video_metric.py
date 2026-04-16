from copy import deepcopy
import torch
from torch import nn
from tqdm import trange
import piqa
import lpips
import numpy as np
import scipy.linalg
from typing import Tuple
import scipy
from torch.cuda.amp import custom_fwd


def batch_forward(batch_size, input1, input2, forward, verbose=False):
    assert input1.shape[0] == input2.shape[0]
    return torch.cat([forward(input1[i: i + batch_size], input2[i: i + batch_size]) for i in trange(0, input1.shape[0], batch_size, disable=not verbose)], dim=0)

import numpy as np
from typing import Tuple, Optional


def symmetrize(mat: np.ndarray) -> np.ndarray:
    """
    将矩阵对称化，消除浮点数误差导致的微小非对称（半正定检查的前提）
    :param mat: 输入矩阵（维度为 [n, n] 的方阵）
    :return: 对称化后的矩阵
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("输入必须是二维方阵！")
    return (mat + mat.T) / 2


def is_positive_semidefinite(
    mat: np.ndarray,
    tol: float = 1e-6
) -> Tuple[bool, np.ndarray, bool]:
    """
    检查对称矩阵的半正定性（核心函数），同时输出是否正定
    :param mat: 输入矩阵（可非对称，函数内自动对称化）
    :param tol: 数值阈值（≤tol的负特征值视为0，处理浮点误差）
    :return: (是否半正定, 所有特征值, 是否正定)
    """
    # 步骤1：强制对称化（半正定仅针对对称矩阵）
    mat_sym = symmetrize(mat)
    
    # 步骤2：计算对称矩阵的特征值（eigvalsh专为对称矩阵优化，更快更稳定）
    try:
        eig_vals = np.linalg.eigvalsh(mat_sym)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"特征值计算失败：{e}")
    
    # 步骤3：判断半正定（所有特征值 ≥ -tol 视为≥0）
    is_semidef = np.all(eig_vals > -tol)
    
    # 步骤4：判断正定（所有特征值 > tol）
    is_pos_def = np.all(eig_vals > tol)
    
    return is_semidef, eig_vals, is_pos_def


def verify_semidefinite_by_quadratic(
    mat: np.ndarray,
    n_samples: int = 100,
    tol: float = 1e-6
) -> Tuple[bool, Optional[float]]:
    """
    抽样验证矩阵的半正定性（二次型法，辅助确认）
    :param mat: 输入矩阵
    :param n_samples: 随机抽样的非零向量数
    :param tol: 数值阈值（≤tol的负数视为0）
    :return: (是否所有抽样二次型≥-tol, 第一个不满足的二次型值/None)
    """
    # 步骤1：对称化矩阵
    mat_sym = symmetrize(mat)
    n = mat_sym.shape[0]
    
    # 步骤2：随机抽样验证二次型
    for _ in range(n_samples):
        # 生成归一化的随机非零向量（避免数值溢出）
        x = np.random.randn(n)
        x_norm = np.linalg.norm(x)
        if x_norm < 1e-10:  # 避免零向量
            x = np.ones(n)
            x_norm = np.linalg.norm(x)
        x = x / x_norm
        
        # 计算二次型 x^T A x
        quad = x.T @ mat_sym @ x
        
        # 检查二次型是否小于阈值
        if quad < -tol:
            return False, quad
    
    # 所有抽样都满足
    return True, None



class Evaluator(nn.Module):
    def __init__(self, i3d_path=None, detector_kwargs=None, max_batchsize=None):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.lpips = lpips.LPIPS(net='vgg')
        self.psnr = piqa.PSNR(epsilon=1e-08, value_range=1.0, reduction='none')
        self.ssim = piqa.SSIM(window_size=11, sigma=1.5, n_channels=3, reduction='none')

        self.i3d_model = torch.jit.load(i3d_path).eval()
        self.max_batchsize = max_batchsize

    def compute_fvd(self, real_feature, gen_feature):
        if real_feature.num_items == 0 or gen_feature.num_items == 0:
            raise ValueError("No data to compute FVD")
        eps = 1e-6
        mu_real, sigma_real = real_feature.get_mean_cov()
        mu_gen, sigma_gen = gen_feature.get_mean_cov()
        # fvd_cov_matrix_check(sigma_semi=deepcopy(sigma_real))
        sigma_real += eps * np.eye(sigma_real.shape[0]) 
        sigma_gen += eps * np.eye(sigma_gen.shape[0]) 
        
        m = np.square(mu_gen - mu_real).sum()
        covmean, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        # fvd_cov_matrix_check(sigma_semi=deepcopy(covmean))

        fid = np.real(m + np.trace(sigma_gen + sigma_real - covmean * 2))
        return float(fid)

    def compute_fvd_from_raw_data(self, real_data=None, gen_data=None):

        detector_kwargs = dict(rescale=True, resize=True,
                               return_features=True)  # Return raw features before the softmax layer.

        mu_real, sigma_real = compute_feature_stats_for_dataset(self.i3d_model, detector_kwargs=detector_kwargs,
                                                                data=real_data).get_mean_cov()

        mu_gen, sigma_gen = compute_feature_stats_for_dataset(self.i3d_model, detector_kwargs=detector_kwargs,
                                                              data=gen_data).get_mean_cov()

        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        return float(fid)

    # custom_fwd: turn off mixed precision to avoid numerical instability during evaluation
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, video_1, video_2):
        # video_1: ground-truth
        # video_2: reconstruction or prediction

        if video_1.shape[0] < video_2.shape[0]:
            B, T, C, H, W = video_1.shape
            t = video_2.shape[0] // B
            video_1 = video_1.repeat([t, 1, 1, 1, 1])

            video_1 = video_1.reshape(-1, C, H, W)
            video_2 = video_2.reshape(-1, C, H, W)

            mse = self.mse(video_1, video_2).mean([1, 2, 3])
            psnr = self.psnr(video_1, video_2)
            ssim = self.ssim(video_1, video_2)
            if self.max_batchsize is not None and video_1.shape[0] > self.max_batchsize:
                lpips = batch_forward(
                    self.max_batchsize,
                    video_1 * 2 - 1, video_2 * 2 - 1,
                    lambda x1, x2: self.lpips(x1, x2).mean((1, 2, 3)),
                )
            else:
                lpips = self.lpips(video_1 * 2 - 1, video_2 * 2 - 1).mean((1, 2, 3))

            # get best of t predictions
            return (
                mse.reshape(t, B, T).mean(-1).min(0).values.mean(),
                psnr.reshape(t, B, T).mean(-1).max(0).values.mean(),
                ssim.reshape(t, B, T).mean(-1).max(0).values.mean(),
                lpips.reshape(t, B, T).mean(-1).min(0).values.mean(),
            )
        else:
            B, T, C, H, W = video_1.shape
            video_1 = video_1.reshape(B * T, C, H, W)
            video_2 = video_2.reshape(B * T, C, H, W)

            return (
                self.mse(video_1, video_2).mean(),
                self.psnr(video_1, video_2).mean(),
                self.ssim(video_1, video_2).mean(),
                self.lpips(video_1 * 2 - 1, video_2 * 2 - 1).mean(),
            )


@torch.no_grad()
def compute_feature_stats_for_dataset(detector, detector_kwargs, data=None):
    stats = FeatureStats(capture_mean_cov=True)

    for i in range(data.size(0)):
        # [batch_size, c, t, h, w]
        images = data[i].permute(0, 2, 1, 3, 4).contiguous()
        if images.shape[1] == 1:
            images = images.repeat([1, 3, *([1] * (images.ndim - 2))])
        features = detector(images, **detector_kwargs)
        stats.append_torch(features)

    return stats


class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x):
        self.append(x.to(torch.float32).cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov


def compute_fvd2(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return float(fid)


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0)  # [d]
    sigma = np.cov(feats, rowvar=False)  # [d, d]

    return mu, sigma

def fvd_cov_matrix_check(
    sigma_semi: FeatureStats,
    feature_dim: int = 400,
    eps: float = 1e-4
) -> None:
    """
    FVD场景专属：模拟协方差矩阵，检查其半正定性/正定性（加正则项前后）
    :param feature_dim: 特征维度（如FVD中视频特征的维度）
    :param eps: 正则项系数（用于将半正定协方差矩阵转为正定）
    """
    print("=" * 60)
    print("FVD场景 - 协方差矩阵半正定性检查")
    print("=" * 60)
    
    # 1. 模拟生成半正定协方差矩阵（样本数<特征维度，模拟实际场景）
    # print("\n【步骤1】生成半正定协方差矩阵（样本数<特征维度）...")
    # sigma_semi = np.random.randn(feature_dim, feature_dim)
    # sigma_semi = sigma_semi @ sigma_semi.T  # 构造半正定矩阵
    
    # 检查半正定性
    is_semidef, eig_vals, is_pos_def = is_positive_semidefinite(sigma_semi)
    min_eig = np.min(eig_vals)
    print(f"原始协方差矩阵 - 是否半正定：{is_semidef}")
    print(f"原始协方差矩阵 - 是否正定：{is_pos_def}")
    print(f"原始协方差矩阵 - 最小特征值：{min_eig:.8f}")
    
    # 二次型验证
    verify_flag, bad_quad = verify_semidefinite_by_quadratic(sigma_semi)
    print(f"原始协方差矩阵 - 二次型验证是否通过：{verify_flag}")
    if not verify_flag:
        print(f"原始协方差矩阵 - 不满足的二次型值：{bad_quad:.8f}")
    
    # 2. 加正则项，将半正定矩阵转为正定
    print("\n【步骤2】加正则项 ε·I，转为正定矩阵...")
    sigma_pos = sigma_semi + eps * np.eye(feature_dim)
    
    # 检查加正则项后的正定性
    is_semidef_reg, eig_vals_reg, is_pos_def_reg = is_positive_semidefinite(sigma_pos)
    min_eig_reg = np.min(eig_vals_reg)
    print(f"加正则项后 - 是否半正定：{is_semidef_reg}")
    print(f"加正则项后 - 是否正定：{is_pos_def_reg}")
    print(f"加正则项后 - 最小特征值：{min_eig_reg:.8f}")
    
    # 二次型验证
    verify_flag_reg, bad_quad_reg = verify_semidefinite_by_quadratic(sigma_pos)
    print(f"加正则项后 - 二次型验证是否通过：{verify_flag_reg}")
    if not verify_flag_reg:
        print(f"加正则项后 - 不满足的二次型值：{bad_quad_reg:.8f}")


# -------------------------- 运行示例 --------------------------
# if __name__ == "__main__":
#     # 配置参数
#     FEATURE_DIM = 1024  # FVD中视频特征的典型维度
#     EPS = 1e-4          # 正则项系数（平衡数值稳定性和结果偏差）
#     TOL = 1e-6          # 数值阈值
    
#     # 运行FVD场景的协方差矩阵检查
#     fvd_cov_matrix_check(feature_dim=FEATURE_DIM, eps=EPS)
    
#     # 额外：自定义矩阵检查示例
#     print("\n" + "=" * 60)
#     print("自定义矩阵半正定性检查")
#     print("=" * 60)
#     # 构造一个半正定矩阵
#     custom_mat = np.array([[1, 2, 3], [2, 5, 7], [3, 7, 10]])
#     is_semidef, eig_vals, is_pos_def = is_positive_semidefinite(custom_mat, tol=TOL)
#     print(f"自定义矩阵：\n{custom_mat}")
#     print(f"是否半正定：{is_semidef}")
#     print(f"特征值：{eig_vals.round(4)}")
#     print(f"是否正定：{is_pos_def}")
