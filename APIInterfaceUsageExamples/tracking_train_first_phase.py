from module.VQPPF import VQPPF
from module.LSTMModule import LSTMModel
import torch



if __name__ == '__main__':
    """
    In the first stage of training, we conduct separate training.
    Need:
        template_patch: Semantically Encoded Static Template Information from Initialization
        search_patch_list: Semantic Information of Search Regions Across Past, Present, and Future
        time_variant_object_state: Semantics of Time-Varying Target States Extracted from Search Regions via Template-Guided Cropping
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
    # Input Examples
    B = 2
    template_patch = torch.randn(B, 768,12, 12)  #
    time_variant_object_state = torch.randn(B, 768,12, 12)
    search_patch_list = [torch.randn(B, 768,24, 24).to(device)] * 3
    template_patch = template_patch.to(device)
    time_variant_object_state = time_variant_object_state.to(device)

    # Input the data into VQPPF
    out_dict = VQPPF_module(template_patch, search_patch_list, time_variant_object_state)

    z_q_t0, z_q_t1, z_q_t2 = out_dict['z_q'][1:]

    lstm_loss = LSTM(z_q_t0.detach(), z_q_t1.detach(), z_q_t2.detach())
    vqppf_loss = out_dict['total_loss']
    print('ok')