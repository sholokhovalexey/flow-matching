

def get_datasets_root(hostname=None):
    if hostname is None:
        import socket
        hostname = socket.gethostname()

    local_paths = {
        "ASUSWHITE": "C:\\Users\\user\\Documents\\datasets\\torch",
        "srv-1gpu": "/home/ubuntu/Documents/datasets/torch",
    }
    return local_paths.get(hostname, "./data")

