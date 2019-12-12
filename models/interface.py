from torch import nn

class ConditionedGenerativeModel(nn.Module):
    '''
    Interface that your models should implement
    '''

    def __init__(self, embd_size):
        '''
        :param embd_size: int, dimension of the conditional embedding
        '''
        self.embd_size = embd_size
        super(ConditionedGenerativeModel, self).__init__()

    def forward(self, imgs, condition_embd):
        '''
        :param imgs: torch.FloatTensor bsize * c * h * w
        :param condition_embd: torch.FloatTensor bsize * embd_size
        :return: outputs : dict of ouputs, this can be {"d_loss" : d_loss, "g_loss" : g_loss"} for a gan
        '''
        raise NotImplementedError

    def likelihood(self, imgs, condition_embd):
        '''
        :param imgs: torch.FloatTensor bsize * c * h * w
        :param condition_embd: torch.FloatTensor bsize * embd_size
        :return: likelihoods : torch.FloatTensor of size bSize, likelihoods of the images conditioned on the captions
        '''
        raise NotImplementedError

    def sample(self, condition_embd):
        '''
        :param condition_embd: torch.FloatTensor bsize * embd_size
        :return: imgs : torch.FloatTensor of size n_imgs * c * h * w
        '''
        raise NotImplementedError