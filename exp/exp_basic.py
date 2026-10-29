"""实验基类：设备选择 + 模型自动发现。

参考 thuml/Time-Series-Library 的 ``exp/exp_basic.py`` 设计：
``models/`` 目录下的每个 ``.py`` 文件都会被自动识别成一个可选模型，文件名就是
``run.py --model`` 的取值，模块里需要提供 ``Model`` 类。

与参考实现一样使用懒加载字典：只有真正被选中的模型才会被 import，因此某个模型缺
依赖（例如 RAT-LLM 需要 ``transformers``）不会影响其它模型启动。
"""

import importlib
import os

import torch


class LazyModelDict(dict):
    """按需 import 的模型字典：``model_dict['LSTM']`` 返回 ``models.LSTM`` 里的 ``Model`` 类。"""

    def __init__(self, model_map):
        super().__init__()
        # {'LSTM': 'models.LSTM', 'CNN': 'models.CNN', ...}
        self.model_map = model_map

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)

        if key not in self.model_map:
            available = ', '.join(sorted(self.model_map)) or '（空）'
            raise KeyError(
                f"未知模型 [{key}]：models/ 下没有 {key}.py；可用模型：{available}"
            )

        module_path = self.model_map[key]
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"导入模型 [{key}] 失败（{module_path}），可能缺少依赖：{exc}"
            ) from exc

        if hasattr(module, 'Model'):
            model_class = module.Model
        elif hasattr(module, key):
            model_class = getattr(module, key)
        else:
            raise AttributeError(
                f"{module_path} 中没有 Model 类（也没有名为 {key} 的类）"
            )

        self[key] = model_class
        return model_class


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = LazyModelDict(self._scan_models_directory())
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _scan_models_directory(self):
        """扫描项目根目录下的 models/，返回 {模型名: 模块路径}。"""
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(root_dir, 'models')

        model_map = {}
        if os.path.isdir(models_dir):
            for filename in sorted(os.listdir(models_dir)):
                # 跳过 __init__.py、以下划线开头的临时文件和非 Python 文件
                if filename.endswith('.py') and not filename.startswith('_'):
                    module_name = filename[:-3]
                    model_map[module_name] = f"models.{module_name}"
        return model_map

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            if torch.cuda.is_available():
                return torch.device('cuda:{}'.format(self.args.gpu))
            print('Warning: 未检测到可用的 CUDA，自动改用 CPU 运行。')
        return torch.device('cpu')

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
