from module.VQPPF import VQPPF
from module.LSTMModule import LSTMModel
import torch



if __name__ == '__main__':
    """
    
    Need:
        Trained VQPPF weight parameters
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Example of VQPPF Architecture Construction
    hidden_dim =768
    n_hidden = 128
    n_residual_hidden = 2
    n_residual_layer = 2
    n_emdedding = 512
    embedding_dim = 64
    beta = 0.25

    VQPPF_module = VQPPF(hidden_dim, n_hidden, n_residual_hidden, n_residual_layer, n_emdedding, embedding_dim, beta)
    VQPPF_module = VQPPF_module.to(device)

    # Example of LSTM Architecture Construction
    input_size = 2304
    hidden_size = 256
    num_layer = 5
    num_class = 2304

    LSTM = LSTMModel(input_size, hidden_size, num_layer, num_class)
    LSTM = LSTM.to(device)

    # Initialization of latent variables for initial impression
    template_patch = torch.randn(1, 768,12, 12)  #
    template_patch = template_patch.to(device)

    zq_c = VQPPF_module.encoderZ(template_patch)

    # Tracking of sequential frames
    search_patch_list = [torch.randn(1, 768,24, 24).to(device)] * 50


    interval_T = 5
    frame_id = 1
    zqd_queue = []
    for search_patch in search_patch_list:  # Just an example. The actual method of collecting information in the search area should be followed
        zq_d_pr = VQPPF_module.encoderX(search_patch)
        if frame_id < interval_T:
            zqd_queue.append(zq_d_pr)
        else:
            zqd_queue = zqd_queue[1:]
            zqd_queue.append(zq_d_pr)
        frame_id += 1

        zq_d_pa = zqd_queue[0]
        zq_d_fu = LSTM.prediction(zq_d_pa, zq_d_pr)  # (1,64,2,2)
        object_vt = VQPPF_module.decodeInference([zq_c, zq_d_pa, zq_d_pr, zq_d_fu])  # (1,768,8,8)

    print('ok')
