########### 할 것
# 1. generator의 input 부분 부터 수정해야한다. 왜? 4x4x512가 아니라 32x32x8을 input으로 시작한다.
# 2. refecltion padding (resblock), no padding (conv), zero padding (generator) 논문 참조 figure18
#    -> convlayer에서 padding을 여러개 처리할 수 있도록 변경
# 3. resblock에는 conv2d, generator resblock에는 modulatedconv2d가 쓰이면 될 것 같음. styledconv는?
#    -> generator 부분 다시 봐야한다.. 아직 잘 모르겟음
# 4. fIGURE 18 보고 256X256으로 재구축 해볼 것
# 5. B.2 R1 reg 읽어보고 적용 필요 -> path length까지 적용해야 할듯..이것도 r1 일종 같음
# 6. linear layer 마지막에 activation 불필요?
# 7. cooor에 왜 g loss가 들어가는지?

import torch
from torch.utils.data import DataLoader
import argparse
import time
from torch import optim
from torch.nn import functional as F
from network import *
from utils import *
from dataset import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100000, help='Iterations')
    parser.add_argument('--size', type=int, default=256, help='Image size')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--fm_U', type=int, default=3, help='feature matching for disc U')
    parser.add_argument('--fm_AB', type=int, default=3, help='feature matching for disc A,B')
    parser.add_argument('--n_downsample', type=int, default=2, help='n_downsample')
    parser.add_argument('--dim', type=int, default=64, help='Encoder dim')
    parser.add_argument('--style_dim', type=int, default=64, help='style_dim')
    parser.add_argument('--n_upsample', type=int, default=2, help='n_upsample')
    parser.add_argument('--n_res', type=int, default=4, help='n_res')
    parser.add_argument('--norm', type=str, default='none', help='norm')
    parser.add_argument('--activ', type=str, default='relu', help='activ')
    parser.add_argument('--pad_type', type=str, default='reflect', help='pad_type')
#     parser.add_argument('--channel_multiplier', type=int, default=1, help='Generator channel multiplier')
    #     parser.add_argument('--texture', type=int, default=2048, help='Generator texture channels')
    #     parser.add_argument('--crop_num', type=int, default=8, help='Patches for each image')
    #     parser.add_argument('--ref_num', type=int, default=4, help='Reference patches')
    #     parser.add_argument('--stylegan_weights', type=str, default='/home/stylegan2-pytorch_original/stylegan2-pytorch/tb_2nd_female_only_ju_freezeD_512_Mixed2_32_55.pt', help='stylegan weights')
    # parser.add_argument('--n_mlp', type=int, default=8, help='Mapping network layers')
    #     parser.add_argument('--d_reg_every', type=int, default=8, help='Lazy R1 regularization interval')
    #     parser.add_argument('--disc_r1', type=float, default=10.0, help='Lazy R1 regularization weight for discriminator')
    #     parser.add_argument('--p_disc_r1', type=float, default=1.0, help='Lazy R1 regularization weight for patch discriminator')
    parser.add_argument('--beta1', type=float, default=0, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.99, help='Beta2 for Adam optimizer')
    #     parser.add_argument('--lazy_r1', type=bool, default=True, help='Lazy R1 regularization')
    # parser.add_argument('--model_dir', type=str, default='model', help='model directory')
    # parser.add_argument('--model_name', type=str, default='female', help='model name.pt')
    parser.add_argument('--print', type=int, default=1, help='loss print interval')
    parser.add_argument('--model_save', type=int, default=1, help='model save interval')
    parser.add_argument('--image_save', type=int, default=1, help='intermediate image save interval')
    parser.add_argument('--image_path', type=str, default='/home/AniGAN/exp/images/',
                        help='intermediate image result')
    parser.add_argument('--dataset_path', type=str, default='/home/sa/dataset_n/', help='dataset directory')
    parser.add_argument('--output_path', type=str, default='/home/AniGAN/exp/checkpoints/',
                        help='output directory')
    parser.add_argument('--resume', type=bool, default=False, help='resume')
    parser.add_argument('--log_name', type=str, default='/home/AniGAN/exp/log.txt', help='log file name')
    parser.add_argument('--num_workers', type=int, default=8, help='workers for dataloader')
    return parser.parse_args()


def Adv_D_loss(real_logit, fake_logit):
    loss = torch.mean(F.softplus(-real_logit)) + torch.mean(F.softplus(fake_logit))
    return loss


def Adv_G_loss(fake_logit):
    loss = torch.mean(F.softplus(-fake_logit))
    return 


def recon_loss(recon, images):
    loss = torch.nn.L1Loss()
    return loss(recon, images)


def train():
    s_time = time.time()
    args = parse_args()

    #     if args.lazy_r1:
    #         d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    #     else:
    #         d_reg_ratio = 1

    device = torch.device("cuda:0" if (torch.cuda.is_available() > 0) else "cpu")
    train_dataset = ImageDataset(args.dataset_path, size=args.size, unpaired=False, mode='train')
    test_dataset = ImageDataset(args.dataset_path, size=args.size, unpaired=False, mode='test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )
    train_iter = iter(train_loader)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        num_workers=1
    )
    test_iter = iter(test_loader)

    style_encoder = StyleEncoder(args.n_downsample, 3, args.dim, args.style_dim, args.norm, args.activ, args.pad_type).to(device)
    content_encoder = ContentEncoder(args.n_downsample, args.n_res, 3, args.dim, args.norm, args.activ, args.pad_type).to(device)
    decoder = Decoder(args.n_upsample, args.dim, output_dim=3, input_dim=256, num_ASC_layers=4, num_FST_blocks=2, activ='relu', pad_type='zero').to(device)
    discriminator_U = Discriminator_U(args.size, norm='in', activation='relu', pad_type='zero').to(device)
    discriminator_A = Discriminator_X(args.size, norm='in', activation='relu', pad_type='zero').to(device)
    discriminator_B = Discriminator_Y(args.size, norm='in', activation='relu', pad_type='zero').to(device)

    activation_U = {}
    activation_A = {}
    activation_B = {}

    def get_activation(name, activation):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    for i in range(args.fm_U):  # 0,1,2
        discriminator_U.blocks[i].register_forward_hook(get_activation(str(i), activation_U))

    for i in range(args.fm_AB):  # 0,1,2
        discriminator_A.blocks[i].register_forward_hook(get_activation(str(i), activation_A))
        discriminator_B.blocks[i].register_forward_hook(get_activation(str(i), activation_B))

    g_optim = optim.Adam(
        list(style_encoder.parameters()) + list(content_encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        betas=(args.beta1, args.beta2)
    )

    d_optim = optim.Adam(
        list(discriminator_A.parameters()) + list(discriminator_B.parameters()) + list(discriminator_U.parameters()),
        lr=args.lr,
        betas=(args.beta1, args.beta2)
    )

    # if args.resume:
    #     print("load model")
    #     model_path = os.path.join(args.output_path, 'model_050000.pt')
    #     ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
    #     iteration = ckpt["iter"]
    #     print(f'from {str(iteration)} resumed...')
    #     encoder_A.load_state_dict(ckpt["encoder_A"])
    #     generator_A.load_state_dict(ckpt["generator_A"])
    #     discriminator_A.load_state_dict(ckpt["discriminator_A"])
    #     patch_discriminator_A.load_state_dict(ckpt["patch_discriminator_A"])
    #     encoder_B.load_state_dict(ckpt["encoder_B"])
    #     generator_B.load_state_dict(ckpt["generator_B"])
    #     discriminator_B.load_state_dict(ckpt["discriminator_B"])
    #     patch_discriminator_B.load_state_dict(ckpt["patch_discriminator_B"])
    #     g_optim.load_state_dict(ckpt["g_optimizer"])
    #     d_optim.load_state_dict(ckpt["d_optimizer"])
    #     print("load model completed")
    # else:
    #     iteration = 1

    # loss_lpips = LPIPS(net_type='alex').to(device).eval()
    # i_loss = id_loss.IDLoss(path='/home/pixel2style2pixel_ada/pretrained_models/model_ir_se50.pth').to(device).eval()

    # load generator
    # print('Loading decoder weights from pretrained!')
    # ckpt = torch.load(args.stylegan_weights)
    # decoder.load_state_dict(ckpt['g_ema'], strict=False)
    #     print(decoder)
    #     if self.opts.learn_in_w:
    #         self.__load_latent_avg(ckpt, repeat=1)
    #     else:
    #         self.__load_latent_avg(ckpt, repeat=18)

    #     def __load_latent_avg(self, ckpt, repeat=None):
    #         if 'latent_avg' in ckpt:
    #             self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
    #             if repeat is not None:
    #                 self.latent_avg = self.latent_avg.repeat(repeat, 1)
    #         else:
    #             self.latent_avg = None

    iteration = 1

    while True:
        try:
            images = next(train_iter)
        except StopIteration:
            print('train dataloader re-initialized')
            train_iter = iter(train_loader)
            images = next(train_iter)
        images_A = images['images_A'].to(device)                                                                         # 1 x 3 x 256 x 256 (FFHQ)
        images_B = images['images_B'].to(device)                                                                         # 1 X 3 X 256 X 256 (Anime)

        #################################### Discriminator ###############################################

        requires_grad(discriminator_A, True)
        requires_grad(discriminator_B, True)
        requires_grad(discriminator_U, True)
        requires_grad(style_encoder, False)
        requires_grad(content_encoder, False)
        requires_grad(decoder, False)

        c_code_A = content_encoder(images_A)                                                                             # 1 x 256 x 64 x 64
        s_code_B = style_encoder(images_B)                                                                               # 1 x 128
        fake_B = decoder(c_code_A, s_code_B)                                                                             # 1 x 3 x 256 x 256

        c_code_B = content_encoder(images_B)                                                                             # 1 x 256 x 64 x 64
        s_code_A = style_encoder(images_A)                                                                               # 1 x 128
        fake_A = decoder(c_code_B, s_code_A)                                                                             # 1 x 3 x 256 x 256

        real_logit_A = discriminator_A(discriminator_U(images_A))
        fake_logit_A = discriminator_A(discriminator_U(fake_A.detach()))

        real_logit_B = discriminator_B(discriminator_U(images_B))
        fake_logit_B = discriminator_B(discriminator_U(fake_B.detach()))

        d_loss = Adv_D_loss(real_logit_A, fake_logit_A) + Adv_D_loss(real_logit_B, fake_logit_B)
        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()


        #################################### Generator #########################################################

        requires_grad(discriminator_A, False)
        requires_grad(discriminator_B, False)
        requires_grad(discriminator_U, False)
        requires_grad(style_encoder, True)
        requires_grad(content_encoder, True)
        requires_grad(decoder, True)

        c_code_A = content_encoder(images_A)  # 1 x 256 x 64 x 64
        s_code_B = style_encoder(images_B)  # 1 x 128
        fake_B = decoder(c_code_A, s_code_B)  # 1 x 3 x 256 x 256

        c_code_B = content_encoder(images_B)  # 1 x 256 x 64 x 64
        s_code_A = style_encoder(images_A)  # 1 x 128
        fake_A = decoder(c_code_B, s_code_A)  # 1 x 3 x 256 x 256

        fake_logit_A = discriminator_A(discriminator_U(fake_A))
        fake_logit_B = discriminator_B(discriminator_U(fake_B))

        adv_g_loss = Adv_G_loss(fake_logit_A) + Adv_G_loss(fake_logit_B)
        g_optim.zero_grad()
        adv_g_loss.backward()
        g_optim.step()


        ################# Feature matching ######################## ------------> Generator? Discriminator??

        ### Domain A
        c_code_A = content_encoder(images_A)  # 1 x 256 x 64 x 64
        s_code_A = style_encoder(images_A)  # 1 x 128
        recon_A = decoder(c_code_A, s_code_A)  # 1 x 3 x 256 x 256

        _ = discriminator_A(discriminator_U(recon_A)) # 2개의 hook이 돌아간다 disc U, disc A 각각
        recon_U_feature_A = activation_U.deepcopy()  # disc U에 fake A 태울 경우
        recon_A_feature_A = activation_A.deepcopy()  # disc A에 fake A 태울 경우

        _ = discriminator_A(discriminator_U(images_A))
        real_U_feature_A = activation_U.deepcopy()
        real_A_feature_A = activation_A.deepcopy()

        for _ in range(args.fm_u):
            fm_loss_UA += recon(recon_U_feature_A[str(i)].mean([2,3]),real_U_feature_A[str(i)].mean([2,3]))
            fm_loss_AA += recon(recon_A_feature_A[str(i)].mean([2,3]),real_A_feature_A[str(i)].mean([2,3]))

        ### Domain B
        c_code_B = content_encoder(images_B)  # 1 x 256 x 64 x 64
        s_code_B = style_encoder(images_B)  # 1 x 128
        recon_B = decoder(c_code_B, s_code_B)  # 1 x 3 x 256 x 256

        _ = discriminator_B(discriminator_U(recon_B))  # 2개의 hook이 돌아간다 disc U, disc A 각각
        recon_U_feature_B = activation_U.deepcopy()  # disc U에 fake A 태울 경우
        recon_B_feature_B = activation_B.deepcopy()  # disc A에 fake A 태울 경우

        _ = discriminator_B(discriminator_U(images_B))
        real_U_feature_B = activation_U.deepcopy()
        real_B_feature_B = activation_B.deepcopy()

        for _ in range(args.fm_u):
            fm_loss_UB += recon(recon_U_feature_B[str(i)].mean([2, 3]), real_U_feature_B[str(i)].mean([2, 3]))
            fm_loss_BB += recon(recon_B_feature_B[str(i)].mean([2, 3]), real_B_feature_B[str(i)].mean([2, 3]))


        fm_loss = fm_loss_UA + fm_loss_AA + fm_loss_UB + fm_loss_BB
        g_optim.zero_grad()
        fm_loss.backward()
        g_optim.step()

        ##### recon loss
        c_code_A = content_encoder(images_A)  # 1 x 256 x 64 x 64
        s_code_A = style_encoder(images_A)  # 1 x 128
        recon_A = decoder(c_code_A, s_code_A)  # 1 x 3 x 256 x 256
        reon_loss_AA = recon(recon_A, images_A)
        g_optim.zero_grad()
        fm_loss.backward()
        g_optim.step()



        if iteration % args.print == 0:
            time_elapsed = timer(s_time, time.time())
            s = f'iteration: {iteration}, d loss: {d_loss:.3f} g_loss: {g_loss:.3f}  adv_g_loss: {adv_g_loss:.3f} rec_g_loss: {rec_g_loss:.3f} time: {time_elapsed}'
            print(s)
            with open(args.log_name, "a") as log_file:
                log_file.write('%s\n' % s)


        if iteration % args.model_save == 0:
            model_save_path = os.path.join(args.output_path, f'model_{str(iteration).zfill(6)}.pt')
            torch.save({
                'iter': iteration,
                'style_encoder': style_encoder.state_dict(),
                'content_encoder': content_encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'discriminator_U': discriminator_U.state_dict(),
                'discriminator_A': discriminator_A.state_dict(),
                'discriminator_B': discriminator_B.state_dict(),
                'g_optimizer': g_optim.state_dict(),
                'd_optimizer': d_optim.state_dict(),
            }, model_save_path
            )
            print(f'model_{str(iteration)} saved')

        if iteration % args.image_save == 0:
            try:
                test_images = next(test_iter)
            except StopIteration:
                test_iter = iter(test_iter)
                test_images = next(test_iter)

            test_images_A = test_images['images_A'].to(device)
            test_images_B = test_images['images_B'].to(device)

            test_s_code_A, test_t_code_A = style_encoder(test_images_A)
            test_s_code_B, test_t_code_B = content_encoder(test_images_B)
            test_recon_A = decoder(test_s_code_A, test_t_code_A)
            test_recon_B = decoder(test_s_code_B, test_t_code_B)
            test_hybrid_A = decoder(test_s_code_B, test_t_code_A)
            test_hybrid_B = decoder(test_s_code_A, test_t_code_B)
            img_save(test_images_A, 'image_A', iteration, args.image_path)
            img_save(test_images_B, 'image_B', iteration, args.image_path)
            img_save(test_recon_A, 'recon_A', iteration, args.image_path)
            img_save(test_hybrid_A, 'hybrid_A', iteration, args.image_path)
            img_save(test_recon_B, 'recon_B', iteration, args.image_path)
            img_save(test_hybrid_B, 'hybrid_B', iteration, args.image_path)

            print(f'images_{str(iteration)} saved')

        #             try:
        #                 val_images = next(val_iter)
        #             except StopIteration:
        #                 val_iter = iter(val_iter)
        #                 val_images = next(val_iter)

        #             val_images = val_images.to(device)
        #             imageA = val_images[0].unsqueeze(0)
        #             imageB = val_images[1].unsqueeze(0)
        #             val_s_code_A, val_t_code_A = encoder(imageA)
        #             val_s_code_B, val_t_code_B = encoder(imageB)
        #             val_recon = generator(val_s_code_A, val_t_code_A)
        #             val_hybrid = generator(val_s_code_A, val_t_code_B)
        #             img_save(imageA, 'imageA', iteration, args.image_path)
        #             img_save(imageB, 'imageB', iteration, args.image_path)
        #             img_save(val_recon, 'recon', iteration, args.image_path)
        #             img_save(val_hybrid, 'hybrid', iteration, args.image_path)
        #             print(f'images_{str(iteration)} saved')

        if iteration >= args.iterations:
            break
        else:
            iteration += 1


if __name__ == '__main__':
    print('train started')
    train()
    print('training finished')










