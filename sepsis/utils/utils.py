class AttrDict(dict):
    """Dictionary subclass whose entries can be accessed by attributes"""
    def __init__(self, *args, **kwargs):
        def from_nested_dict(data):
            """ Construct nested AttrDicts from nested dictionaries. """
            if not isinstance(data, dict):
                return data
            else:
                return AttrDict({key: from_nested_dict(data[key])
                                    for key in data})

        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

        for key in self.keys():
            self[key] = from_nested_dict(self[key])


def delta_features():
    """Computes delta features.

    Returns
    -------

    """
    pass



def filter_topn():
    """"""
    pass

def filter_dataset(df):
    """"""
    pass
