import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))     # this is needed to import 'our_modules' from the parent directory

import our_modules
from our_modules import autoencoder

### This is just an utility function to make sure the correct model is built
# The actual implementations of the encoder, downsampler and decoder, can be found respectively in our_modules/{encoder, autoencoder, decoder}

def dgpp(remove_point_num=256, pretrained=False):
    model = autoencoder.build_model(enc_type='dg',
                    encoding_length=512,
                    dec_type='ppd',
                    method='missing',
                    remove_point_num=remove_point_num)
    if pretrained:
        pass    #todo

    return model