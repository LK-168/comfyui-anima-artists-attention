"""画师链解析工具。"""

MAX_ARTISTS = 32


def split_artist_chain(chain):
    """将画师字符串拆分为列表。支持逗号、中文逗号、换行分隔。"""
    if not chain:
        return []
    s = str(chain).replace("，", ",").replace("\n", ",").replace("\r", ",")
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def parse_artist_weights(names):
    """
    解析画师名和权重。
    支持格式: 'name', '(name:1.2)', '(@name:0.8)'
    返回: (names, weights)
    """
    parsed, weights = [], []
    for name in names:
        if ":" in name and name.rfind(":") > 0:
            parts = name.rsplit(":", 1)
            try:
                w = float(parts[1].strip().rstrip(")"))
                n = parts[0].strip().lstrip("(")
                parsed.append(n)
                weights.append(w)
            except ValueError:
                parsed.append(name)
                weights.append(1.0)
        else:
            parsed.append(name)
            weights.append(1.0)
    return parsed, weights
