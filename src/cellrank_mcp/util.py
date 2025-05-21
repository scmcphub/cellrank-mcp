import inspect
import os
from pathlib import Path
from scmcp_shared.util import get_env

def set_fig_path(func,  **kwargs):
    fig_dir = Path(os.getcwd()) / "figures"

    kwargs.pop("save", None)
    kwargs.pop("show", None)
    args = []
    for k,v in kwargs.items():
        if isinstance(v, (tuple, list, set)):
            args.append(f"{k}-{'-'.join([str(i) for i in v])}")
        else:
            args.append(f"{k}-{v}")
    args_str = "_".join(args)
    if func == "scvelo_projection":
        old_path = fig_dir / f"scvelo_{kwargs['kernel']}.png"
        fig_path = fig_dir / f"{func}_{args_str}.png"
    else:
        if (fig_dir / f"{func}_.png").is_file():
            old_path = fig_dir / f"{func}_.png"
        else:
            old_path = fig_dir / f"{func}.png"
        fig_path = fig_dir / f"{func}_{args_str}.png"
    try:
        os.rename(old_path, fig_path)
        return fig_path
    except FileNotFoundError:
        print(f"The file {old_path} does not exist")
    except FileExistsError:
        print(f"The file {fig_path} already exists")
    except PermissionError:
        print("You don't have permission to rename this file")
    transport = get_env("TRANSPORT") 
    if transport == "stdio":
        return fig_path
    else:
        host = get_env("HOST")
        port = get_env("PORT")
        fig_path = f"http://{host}:{port}/figures/{Path(fig_path).name}"
        return fig_path