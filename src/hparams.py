def encoder_decoder_params(dataset):
    if dataset in ['mnist', 'fashion_mnist', 'cifar10']:
        encoder_params=[]
        encoder_params.append([32, 4, (2,2)]) #Filter, kernelsize, stride
        encoder_params.append([64, 4, (2,2)])
        encoder_params.append([128, 4, (2,2)])
        decoder_params=[]
        decoder_params.append([128, 4, (2,2)])
        decoder_params.append([64, 4, (2,2)])
        decoder_params.append([32, 4, (1,1)])
    elif dataset in ['celebA']:
        encoder_params=[]
        encoder_params.append([64, 4, (2,2)])
        encoder_params.append([128, 4, (2,2)])
        encoder_params.append([256, 4, (2,2)])
        decoder_params=[]
        decoder_params.append([256, 4, (2,2)])
        decoder_params.append([128, 4, (2,2)])
        decoder_params.append([64, 4, (2,2)])
    elif dataset in ['dsprites']:
        encoder_params=[]
        encoder_params.append([32, 4, (2,2)])
        encoder_params.append([64, 4, (2,2)])
        decoder_params=[]
        decoder_params.append([64, 4, (2,2)])
        decoder_params.append([32, 4, (2,2)])
    return encoder_params, decoder_params