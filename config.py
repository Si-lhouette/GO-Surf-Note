import os
import yaml


# load config路径下两部分的config文件: base_config & config
# 并将拼合的config文件写入logs路径下
def load_config(exp_name=None, scene="", use_config_snapshot=False):
    # os.path.join函数可以接受任意数量的输入变量,这个函数会将所有输入变量拼接成一个路径，并且添加斜杠作为间隔
    log_dir = os.path.join(os.path.dirname(__file__), "logs", scene) 

    if exp_name:
        log_dir = os.path.join(log_dir, exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # split函数会将一个字符串根据指定的分隔符进行分割，并返回分割后的多个子字符串组成的列表
    config_file = scene.split("/")[-1] + ".yaml"  
    
    if use_config_snapshot:  # log complete config file, i.e. from log_dir
        config_path = os.path.join(log_dir, config_file)
        
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:  # load from configs/xxxx.yaml, i.e. start a new training
        base_config_path = os.path.join(os.path.dirname(__file__),  "configs/base.yaml")
        config_path = os.path.join(os.path.dirname(__file__),  "configs", config_file)
        
        # load basic config
        with open(base_config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
        # load specific config
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = {**config, **yaml.load(f, Loader=yaml.FullLoader)}

        # 将字典对象config写入yaml文件f
        with open(os.path.join(log_dir, config_file), "w") as f:
            yaml.dump(config, f, indent=4, default_flow_style=None, sort_keys=False)
        
    # 加入两个新参数
    config["log_dir"] = log_dir
    config["checkpoints_dir"] = os.path.join(log_dir, "checkpoints/")

    if not os.path.exists(config["checkpoints_dir"]):
        os.makedirs(config["checkpoints_dir"])

    return config