

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


from torch.utils.data import DataLoader
import argparse
import time
from torch import optim
from torch.nn import functional as F
from network import *
from utils import *
from layers import *
from dataset import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100000, help='Iterations')
    parser.add_argument('--size', type=int, default=256, help='Image size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--in_channels', type=int, default=32, help='Encoder in-channels')
    parser.add_argument('--channel_multiplier', type=int, default=1, help='Generator channel multiplier')
    parser.add_argument('--texture', type=int, default=2048, help='Generator texture channels')
    parser.add_argument('--crop_num', type=int, default=8, help='Patches for each image')
    parser.add_argument('--ref_num', type=int, default=4, help='Reference patches')
    # parser.add_argument('--n_mlp', type=int, default=8, help='Mapping network layers')
    parser.add_argument('--d_reg_every', type=int, default=8, help='Lazy R1 regularization interval')
    parser.add_argument('--disc_r1', type=float, default=10.0, help='Lazy R1 regularization weight for discriminator')
    parser.add_argument('--p_disc_r1', type=float, default=1.0, help='Lazy R1 regularization weight for patch discriminator')
    parser.add_argument('--beta1', type=float, default=0, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.99, help='Beta2 for Adam optimizer')
    parser.add_argument('--lazy_r1', type=bool, default=True, help='Lazy R1 regularization')
    # parser.add_argument('--model_dir', type=str, default='model', help='model directory')
    # parser.add_argument('--model_name', type=str, default='female', help='model name.pt')
    parser.add_argument('--print', type=int, default=500, help='loss print interval')
    parser.add_argument('--model_save', type=int, default=2000, help='model save interval')
    parser.add_argument('--image_save', type=int, default=100, help='intermediate image save interval')
    parser.add_argument('--image_path', type=str, default='/home/sa/images2', help='intermediate image result')
    parser.add_argument('--dataset_path', type=str, default='/home/sa/dataset/', help='dataset directory')
    parser.add_argument('--output_path', type=str, default='/home/sa/checkpoint/', help='output directory')
    parser.add_argument('--resume_path', type=bool, default=True, help='resume directory')    
    parser.add_argument('--log_name', type=str, default='/home/sa/log.txt', help='log file name')
    parser.add_argument('--num_workers', type=int, default=8, help='workers for dataloader')
    return parser.parse_args()


def Adv_D_loss(real_logit, fake_logit):
    loss = torch.mean(F.softplus(-real_logit)) + torch.mean(F.softplus(fake_logit))
    return loss

def Adv_G_loss(fake_logit):
    loss = torch.mean(F.softplus(-fake_logit))
    return loss

def recon_loss(recon, images):
    loss = torch.nn.L1Loss()
    return loss(recon, images)

def train():

    s_time = time.time()
    args = parse_args()

    if args.lazy_r1:
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    else:
        d_reg_ratio = 1

    device = torch.device("cuda:0" if (torch.cuda.is_available() > 0) else "cpu")
    train_dataset = ImageDataset(args.dataset_path, size=args.size, mode='train')
    val_dataset = ImageDataset(args.dataset_path, size=args.size, mode='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )
    train_iter = iter(train_loader)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        num_workers=1
    )
    val_iter = iter(val_loader)

    # print('here2')
    encoder = Encoder(args.in_channels)
    encoder = nn.DataParallel(encoder).to(device)    
     
    patch_discriminator = PatchDiscriminator(args.in_channels)
    patch_discriminator = nn.DataParallel(patch_discriminator).to(device)    
        
    generator = Generator(args.texture)
    generator = nn.DataParallel(generator).to(device)    
    
    discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier)
    discriminator = nn.DataParallel(discriminator).to(device)    
    

#     encoder = Encoder(args.in_channels).to(device)    
#     patch_discriminator = PatchDiscriminator(args.in_channels).to(device)
#     generator = Generator(args.texture).to(device)
#     discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)
    # print('here')

    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
                # print(p.name, nn)
                # print(p)
            pp += nn
        # print('finished')
        return pp
    # print('encoder', get_n_params(encoder))
    # print('patch_discriminator', get_n_params(patch_discriminator))
    # print('generator', get_n_params(generator))
    # print('discriminator', get_n_params(discriminator))

    g_optim = optim.Adam(
        list(generator.parameters()) + list(encoder.parameters()),
        lr=args.lr,
        betas=(args.beta1, args.beta2)
    )

    d_optim = optim.Adam(
        list(discriminator.parameters()) + list(patch_discriminator.parameters()),
        lr=args.lr * d_reg_ratio,
        betas=(args.beta1 ** d_reg_ratio, args.beta2 ** d_reg_ratio),
    )


    if args.resume_path:
        print("load model")
        model_path = '/home/sa/checkpoint/model_034000.pt'
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        iteration = ckpt["iter"]
        print(f'from {str(iteration)} resumed...')
        encoder.load_state_dict(ckpt["encoder"])
        generator.load_state_dict(ckpt["generator"])
        discriminator.load_state_dict(ckpt["discriminator"])
        patch_discriminator.load_state_dict(ckpt["patch_discriminator"])
        g_optim.load_state_dict(ckpt["g_optimizer"])
        d_optim.load_state_dict(ckpt["d_optimizer"])
        print("load model completed")
    else:
        iteration = 1

    while True:
        try:
            images = next(train_iter)
        except StopIteration:
            print('train dataloader re-initialized')
            train_iter = iter(train_loader)
            images = next(train_iter)
        images = images.to(device)
        images = images[torch.randperm(images.size(0))]
        mid_pos = images.size(0) // 2
        images_A = images[:mid_pos]                                                                         # for recon
        images_B = images[mid_pos:]                                                                         # for hybrid

        #################################### Discriminator ###############################################

        requires_grad(discriminator, True)
        requires_grad(patch_discriminator, True)
        requires_grad(generator, False)
        requires_grad(encoder, False)

        # encoding
        s_code_A, t_code_A = encoder(images_A)
        s_code_B, t_code_B = encoder(images_B)
        # print('s_code_A mean', s_code_A.mean())
        # print('t_code_A mean', t_code_A.mean())
        # print('s_code_B mean', s_code_B.mean())
        # print('t_code_B mean', t_code_B.mean())

        # reconstruction & hybrid
        recon = generator(s_code_A, t_code_A)                                                              # recon N//2 images (batch)
        hybrid = generator(s_code_A, t_code_B)                                                             # hybrid N//2 images
        fake = torch.cat([hybrid, recon], dim=0)

        # print('recon', recon.size())
        # print('hybrid', hybrid.size())

        # logit
        # print('fake', fake.size())
        # print('fake', fake.size())

        # print('images mean', images.mean())
        # print('fake mean', fake.mean())
        # print('hybrid mean', hybrid.mean())
        # print('recon mean', recon.mean())

        real_logit = discriminator(images)                                                                                   # fake logit
        fake_logit = discriminator(fake.detach())                                                                   # fake logit
        # print('real_logit', real_logit.mean())
        # print('fake_logit', fake_logit.mean())

        # patch logit
        fake_patches = patch_generation(hybrid, 1, args.crop_num)                                          # 20 x 8 x 1 x 3 x 128 x 128 (# 각 패치마다 prediction 일어나야 한다. (8N prediction) (여기서 N = 40))
        real_patches = patch_generation(images_B, 1, args.crop_num)
        b, p, r, c, h, w = list(fake_patches.size())
        fake_patches = fake_patches.view(-1, c, h, w)                                                      # (20 x 8) x 3 x 128 x 128
        real_patches = real_patches.view(-1, c, h, w)                                                      # (20 x 8) x 3 x 128 x 128
        # import pickle
        # with open("fake_p.pickle", "wb") as fw:
        #     pickle.dump(fake_patches, fw)
        # with open("real_p.pickle", "wb") as fw:
        #     pickle.dump(real_patches, fw)
        # with open("real_img.pickle", "wb") as fw:
        #     pickle.dump(images_B, fw)

        # print('fake_patches', fake_patches.size())
        # print('real_patches', real_patches.size())
        ref_patches = patch_generation(images_B, args.ref_num, args.crop_num)                              # 20 x 8 x 4 x 3 x 128 x 128
        # print('ref_pathces', ref_patches.size())
        ref_patches = ref_patches.view(-1, c, h, w)
        # print('ref_pathces', ref_patches.size())
        # (20 x 8 x 4) x 3 x 128 x 128
        # print('ref_patches', ref_patches.size())
        # print('ref_patches', ref_patches.size())
        # print('fake_patches', fake_patches.size())
        p_fake_logit = patch_discriminator(ref_patches, fake_patches.detach(), args.ref_num)                        # (20 x 8) x 1
        p_real_logit = patch_discriminator(ref_patches, real_patches, args.ref_num)                        # (20 x 8) x 1

        ## Discriminator update
        d_loss = Adv_D_loss(real_logit, fake_logit) + Adv_D_loss(p_real_logit, p_fake_logit)

        # print('loss1', Adv_D_loss(real_logit, fake_logit))
        # print('loss2', Adv_D_loss(p_real_logit, p_fake_logit))

        d_optim.zero_grad()
        # d_loss.backward()
        d_loss.backward()
        d_optim.step()

        if iteration % args.d_reg_every == 0:
            images.requires_grad = True                                                                                 # gradient penalty에서는 image에 대한 grad값이 필요 하므로, true가 되어야 한다.
            real_logit = discriminator(images)
            gp = gradient_penalty(real_logit, images)

            real_patches.requires_grad = True
            p_real_logit = patch_discriminator(ref_patches, real_patches, args.ref_num)
            p_gp = gradient_penalty(p_real_logit, real_patches)

            r1_loss = args.disc_r1 * gp * args.d_reg_every
            p_r1_loss = args.p_disc_r1 * p_gp * args.d_reg_every
            d_optim.zero_grad()
            (r1_loss + p_r1_loss).backward()
            d_optim.step()

            images.requires_grad = False                                                                                # images를 다시 False로 원복 시켜야 한다. 만약 이게 없다면.. images_A, images_B가 requires grad를 가지게 된다. g loss backward 시에 RuntimeError: No grad accumulator for a saved leaf! 발생
            real_patches.requires_grad = False

        #################################### Generator #########################################################

        requires_grad(discriminator, False)
        requires_grad(patch_discriminator, False)
        requires_grad(generator, True)
        requires_grad(encoder, True)

        # print('x before', get_tensor_info(images_A))
        # images.requires_grad = False                                                                                     #images를 다시 False로 원복 시켜야 한다. 만약 이게 없다면.. images_A, images_B가 requires grad를 가지게 된다. g loss backward 시에 RuntimeError: No grad accumulator for a saved leaf! 발생
        # real_patches.requires_grad = False
        # print('x after', get_tensor_info(images_A))

        s_code_A, t_code_A = encoder(images_A)
        s_code_B, t_code_B = encoder(images_B)
        # print('s_code_A mean', s_code_A.mean())
        # print('t_code_A mean', t_code_A.mean())
        # print('s_code_B mean', s_code_B.mean())
        # print('t_code_B mean', t_code_B.mean())

        # reconstruction & hybrid
        recon = generator(s_code_A, t_code_A)                                                                             # recon N//2 images (batch)
        hybrid = generator(s_code_A, t_code_B)                                                                             # hybrid N//2 images
        # print('recon', recon.size())
        # print('hybrid', hybrid.size())

        # logit
        # fake = torch.cat([hybrid, recon], dim=0)
        # print('fake', fake.size())
        # print('fake', fake.size())

        # print('images mean', images.mean())
        # print('fake mean', fake.mean())
        # print('hybrid mean', hybrid.mean())
        # print('recon mean', recon.mean())

        # real_logit = discriminator(images)                                                                                   # fake logit
        # fake_logit = discriminator(fake.detach())                                                                   # fake logit
#         fake_recon_logit = discriminator(recon.detach())                                                               
        fake_recon_logit = discriminator(recon)                                                               
        # fake logit
#         fake_hybrid_logit = discriminator(hybrid.detach())                                                             
        fake_hybrid_logit = discriminator(hybrid)                                                             
        # fake logit

        # print('real_logit', real_logit.mean())
        # print('fake_logit', fake_logit.mean())

        # patch logit
        fake_patches = patch_generation(hybrid, 1, args.crop_num)                                          # 20 x 8 x 1 x 3 x 128 x 128 (# 각 패치마다 prediction 일어나야 한다. (8N prediction) (여기서 N = 40))
        # real_patches = patch_generation(images_B, 1, args.crop_num)
        # b, p, r, c, h, w = list(fake_patches.size())
        #
        fake_patches = fake_patches.view(-1, c, h, w)                                                      # (20 x 8) x 3 x 128 x 128
        # real_patches = real_patches.view(-1, c, h, w)                                                      # (20 x 8) x 3 x 128 x 128
        # import pickle
        # with open("data.pickle", "wb") as fw:
        #     pickle.dump(real_patches, fw)

        # print('fake_patches', fake_patches.size())
        # print('real_patches', real_patches.size())
        ref_patches = patch_generation(images_B, args.ref_num, args.crop_num)                              # 20 x 8 x 4 x 3 x 128 x 128
        ref_patches = ref_patches.view(-1, c, h, w)          ## 여기 무조건 찍어봐야해.....  
    # (20 x 8 x 4) x 3 x 128 x 128
        # print('ref_patches', ref_patches.size())
        # print('ref_patches', ref_patches.size())
        # print('fake_patches', fake_patches.size())
        p_fake_logit = patch_discriminator(ref_patches, fake_patches, args.ref_num)                        # 
#         p_fake_logit = patch_discriminator(ref_patches, fake_patches.detach(), args.ref_num)                        # (20 x 8) x 1
        # p_real_logit = patch_discriminator(ref_patches, real_patches, args.ref_num)                        # (20 x 8) x 1

        ## Generator update
        g_loss = recon_loss(recon, images_A) + 0.5 * Adv_G_loss(fake_recon_logit) + 0.5 * Adv_G_loss(fake_hybrid_logit) + Adv_G_loss(p_fake_logit)
        # g_loss = recon_loss(recon, images_A) + Adv_G_loss(fake_logit) #+ Adv_G_loss(p_fake_logit)
        # print('gloss1', recon_loss(recon, images_A))
        # print('gloss2', Adv_G_loss(fake_logit))
        # print('gloss3', Adv_G_loss(p_fake_logit))
        #
        # requires_grad(discriminator, False)
        # requires_grad(patch_discriminator, False)
        # requires_grad(generator, True)
        # requires_grad(encoder, True)

        # images.requires_grad = False

        g_optim.zero_grad()
        g_loss.backward()
        # g_loss.backward(retain_graph= True)
        g_optim.step()
        # print('generator mean', generator.weight.grad.mean())

        if iteration % args.print == 0:
            time_elapsed = timer(s_time, time.time())
            s = f'iteration: {iteration}, d loss: {d_loss:.3f} g_loss: {g_loss:.3f}  recon_loss: {recon_loss(recon, images_A):.3f} time: {time_elapsed}'
            print(s)
            with open(args.log_name, "a") as log_file:
                log_file.write('%s\n' % s)

        # print('disc', discriminator.parameters())
        # print('gen', generator.parameters())

        # val = 12.3
        # print(f'{val:.2f}')
        # print(f'{val:.5f}')
        # # Print log
        # sys.stdout.write(
        #     f"\r[iteration {iter:d}/{args.iterations:d}] [D adv loss: {d_adv_loss.item():f}, Patch D adv loss: {p_d_adv_loss.item():f}] [G adv loss: {g_adv_loss.item():f}, recon loss: {recon_loss.item():f}, Patch G adv loss: {p_g_adv_loss.item():f}]"
        # )

        if iteration % args.model_save == 0:
            model_save_path = os.path.join(args.output_path, f'model_{str(iteration).zfill(6)}.pt')
            torch.save({
                'iter': iteration,
                'generator': generator.state_dict(),
                'encoder': encoder.state_dict(),
                'discriminator': discriminator.state_dict(),
                'patch_discriminator': patch_discriminator.state_dict(),
                'g_optimizer': g_optim.state_dict(),
                'd_optimizer': d_optim.state_dict(),
            }, model_save_path
            )
            print(f'model_{str(iteration)} saved')

        if iteration % args.image_save == 0:
            try:
                imageA = next(val_iter)
                imageB = next(val_iter)
            except StopIteration:
                val_iter = iter(val_iter)
                imageA = next(val_iter)
                imageB = next(val_iter)

            imageA = imageA.to(device)
            imageB = imageB.to(device)
            val_s_code_A, val_t_code_A = encoder(imageA)
            val_s_code_B, val_t_code_B = encoder(imageB)
            val_recon = generator(val_s_code_A, val_t_code_A)
            val_hybrid = generator(val_s_code_A, val_t_code_B)
            img_save(imageA, 'imageA', iteration, args.image_path)
            img_save(imageB, 'imageB', iteration, args.image_path)
            img_save(val_recon, 'recon', iteration, args.image_path)
            img_save(val_hybrid, 'hybrid', iteration, args.image_path)
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










