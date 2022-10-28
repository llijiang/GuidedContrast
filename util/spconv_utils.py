try:
    import spconv.pytorch as spconv  # spconv 2.x
except:
    import spconv    # spconv 1.x


def replace_feature(out, new_features):
    if 'replace_feature' in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out
