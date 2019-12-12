from utils.pixelcnnpp_utils import *
import pdb
from torch.nn.utils import weight_norm as wn
from tqdm import tqdm
from models.interface import ConditionedGenerativeModel


class ConditionalPixelCNNpp(ConditionedGenerativeModel):
    def __init__(self, embd_size, img_shape, nr_resnet=5, nr_filters=80, nr_logistic_mix=10):
        super(ConditionalPixelCNNpp, self).__init__(embd_size)
        self.pixel_cnn_model = PixelCNNpp(nr_resnet=nr_resnet, nr_filters=nr_filters, nr_logistic_mix=nr_logistic_mix,
                                          embedding_size=embd_size,
                                          input_channels=img_shape[0])
        self.n_logistic_mix = nr_logistic_mix
        self.img_shape = img_shape
        self.loss_function = discretized_mix_logistic_loss
        self.sample_operation = sample_from_discretized_mix_logistic

    def forward(self, imgs, captions_embd):
        '''
        :param imgs: torch.FloatTensor bsize * c * h * w
        :param captions_embd: torch.FloatTensor bsize * embd_size
        :return: outputs : dict of ouputs, this can be {"d_loss" : d_loss, "g_loss" : g_loss"} for a gan
        '''

        bsize, c, h, w = imgs.size()
        # rescaling between [-1, 1]
        x = imgs
        # import pdb;
        # pdb.set_trace()
        model_output = self.pixel_cnn_model(x, captions_embd)

        loss = self.loss_function(imgs, model_output, nmix=self.n_logistic_mix)
        loss_bpd = -loss.sum() / (h * w * bsize * c) #.view(bsize, -1).mean(dim=1)
        loss_by_sample = -loss.view(bsize, -1).sum(dim=1) / (h * w * c)
        outputs = {"loss": loss_bpd,
                   "loss_by_sample": loss_by_sample,
                   "log_likelihood": None,
                   "log_probs": model_output}  
        return outputs

    def likelihood(self, imgs, captions_embd):
        '''
        :param imgs: torch.FloatTensor bsize * c * h * w
        :param captions_embd: torch.FloatTensor bsize * embd_size
        :return: likelihoods : torch.FloatTensor of size bSize, likelihoods of the images conditioned on the captions
        '''
        return None

    def sample(self, captions_embd):
        '''
        :param captions_embd: torch.FloatTensor bsize * embd_size
        :return: imgs : torch.FloatTensor of size n_imgs * c * h * w
        '''

        self.pixel_cnn_model.eval()
        bsize, channels, h, w = [captions_embd.size(0)] + list(self.img_shape)
        data = torch.zeros((bsize, channels, h, w), dtype=torch.float32, device=captions_embd.device,
                           requires_grad=False)
        with torch.no_grad():
            for i in tqdm(range(h)):
                for j in range(w):
                    out = self.pixel_cnn_model(data, captions_embd, sample=True)
                    out_sample = self.sample_operation(out, self.pixel_cnn_model.nr_logistic_mix)
                    data[:, :, i, j] = out_sample[:, :, i, j]
        data = (data + 1) / 2
        return data


class PixelCNNpp(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10, embedding_size=768, input_channels=3):
        super(PixelCNNpp, self).__init__()

        self.resnet_nonlinearity = lambda x: concat_elu(x)
        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters, embedding_size,
                                                             self.resnet_nonlinearity) for i in range(3)])

        self.up_layers = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters, embedding_size,
                                                         self.resnet_nonlinearity) for _ in range(3)])

        self.downsize_u_stream = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters,
                                                                    stride=(2, 2)) for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters,
                                                                           nr_filters, stride=(2, 2)) for _ in
                                                 range(2)])

        self.upsize_u_stream = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters,
                                                                    stride=(2, 2)) for _ in range(2)])

        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters,
                                                                           nr_filters, stride=(2, 2)) for _ in
                                               range(2)])

        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2, 3),
                                          shift_output_down=True)

        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters,
                                                          filter_size=(1, 3), shift_output_down=True),
                                      down_right_shifted_conv2d(input_channels + 1, nr_filters,
                                                                filter_size=(2, 1), shift_output_right=True)])

        num_mix = 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None

    def forward(self, x, h, sample=False):
        # similar as done in the tf repo :
        if (self.init_padding is None or (x.size(0) != self.init_padding.size(0))) and (not sample):
            xs = [int(y) for y in x.size()]
            padding = torch.ones(xs[0], 1, xs[2], xs[3], requires_grad=False)
            self.init_padding = padding.to(x.device)

        if sample:
            xs = [int(y) for y in x.size()]
            padding = torch.ones(xs[0], 1, xs[2], xs[3], requires_grad=False)
            padding = padding.to(x.device)
            x = torch.cat((x, padding), 1)

        ###      UP PASS    ###
        x = x if sample else torch.cat((x, self.init_padding), 1)
        u_list = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1], cond_embedding=h)
            u_list += u_out
            ul_list += ul_out

            if i != 2:
                # downscale (only twice)
                u_list += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        ###    DOWN PASS    ###
        u = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list, cond_embedding=h)

            # upscale (only twice)
            if i != 2:
                u = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out


class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, embedding_size, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d, embedding_size,
                                                    resnet_nonlinearity, skip_connection=0)
                                       for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d, embedding_size,
                                                     resnet_nonlinearity, skip_connection=1)
                                        for _ in range(nr_resnet)])

    def forward(self, u, ul, cond_embedding):
        u_list, ul_list = [], []

        for i in range(self.nr_resnet):
            u = self.u_stream[i](u, conditional_embedding=cond_embedding)
            ul = self.ul_stream[i](ul, a=u, conditional_embedding=cond_embedding)
            u_list += [u]
            ul_list += [ul]

        return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, embedding_size, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d, embedding_size,
                                                    resnet_nonlinearity, skip_connection=1)
                                       for _ in range(nr_resnet)])

        # stream from pixels above and to the left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d, embedding_size,
                                                     resnet_nonlinearity, skip_connection=2)
                                        for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list, cond_embedding):
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u, conditional_embedding=cond_embedding, a=u_list.pop())
            ul = self.ul_stream[i](ul, conditional_embedding=cond_embedding, a=torch.cat((u, ul_list.pop()), 1))

        return u, ul


class nin(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(nin, self).__init__()
        self.lin_a = wn(nn.Linear(dim_in, dim_out))
        self.dim_out = dim_out

    def forward(self, x):
        og_x = x
        # assumes pytorch ordering
        """ a network in network layer (1x1 CONV) """

        x = x.permute(0, 2, 3, 1)
        shp = [int(y) for y in x.size()]
        out = self.lin_a(x.contiguous().view(shp[0] * shp[1] * shp[2], shp[3]))
        shp[-1] = self.dim_out
        out = out.view(shp)
        return out.permute(0, 3, 1, 2)


class down_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 3), stride=(1, 1),
                 shift_output_down=False, norm='weight_norm'):
        super(down_shifted_conv2d, self).__init__()

        assert norm in [None, 'batch_norm', 'weight_norm']
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride)
        self.shift_output_down = shift_output_down
        self.norm = norm
        self.pad = nn.ZeroPad2d((int((filter_size[1] - 1) / 2),  # pad left
                                 int((filter_size[1] - 1) / 2),  # pad right
                                 filter_size[0] - 1,  # pad top
                                 0))  # pad down

        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_down:
            self.down_shift = lambda x: down_shift(x, pad=nn.ZeroPad2d((0, 0, 1, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.down_shift(x) if self.shift_output_down else x


class down_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 3), stride=(1, 1)):
        super(down_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size, stride,
                                            output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        return x[:, :, :(xs[2] - self.filter_size[0] + 1),
               int((self.filter_size[1] - 1) / 2):(xs[3] - int((self.filter_size[1] - 1) / 2))]


class down_right_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 2), stride=(1, 1),
                 shift_output_right=False, norm='weight_norm'):
        super(down_right_shifted_conv2d, self).__init__()

        assert norm in [None, 'batch_norm', 'weight_norm']
        self.pad = nn.ZeroPad2d((filter_size[1] - 1, 0, filter_size[0] - 1, 0))
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride=stride)
        self.shift_output_right = shift_output_right
        self.norm = norm

        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_right:
            self.right_shift = lambda x: right_shift(x, pad=nn.ZeroPad2d((1, 0, 0, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.right_shift(x) if self.shift_output_right else x


class down_right_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 2), stride=(1, 1),
                 shift_output_right=False):
        super(down_right_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size,
                                            stride, output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        x = x[:, :, :(xs[2] - self.filter_size[0] + 1):, :(xs[3] - self.filter_size[1] + 1)]
        return x


'''
skip connection parameter : 0 = no skip connection 
                            1 = skip connection where skip input size === input size
                            2 = skip connection where skip input size === 2 * input size
'''


class gated_resnet(nn.Module):
    def __init__(self, num_filters, conv_op, conditional_embedding_size, nonlinearity=concat_elu, skip_connection=0):
        super(gated_resnet, self).__init__()
        self.skip_connection = skip_connection
        self.nonlinearity = nonlinearity
        self.conv_input = conv_op(2 * num_filters, num_filters)  # cuz of concat elu

        if skip_connection != 0:
            self.nin_skip = nin(2 * skip_connection * num_filters, num_filters)

        self.conditioning_projection = nn.Linear(conditional_embedding_size, 2 * num_filters, bias=False)
        self.dropout = nn.Dropout2d(0.5)
        self.n_filters = num_filters
        self.conv_out = conv_op(2 * num_filters, 2 * num_filters)

    def forward(self, og_x, conditional_embedding, a=None):
        x = self.conv_input(self.nonlinearity(og_x))
        if a is not None:
            x += self.nin_skip(self.nonlinearity(a))
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        embedding_projected = self.conditioning_projection(conditional_embedding)
        x = x + embedding_projected.view(-1, 2 * self.n_filters, 1, 1)
        a, b = torch.chunk(x, 2, dim=1)
        c3 = a * torch.sigmoid(b)
        return og_x + c3
