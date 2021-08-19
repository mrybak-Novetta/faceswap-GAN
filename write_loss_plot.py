import os
import matplotlib.pyplot as plt
import pickle
# loss_file = '/home/cvip/faceswap-GAN/vegas_new_driving_loss_dict_new.pkl'
# loss_file = '/home/cvip/faceswap-GAN/vegas_new_driving_fine_tune_loss_dict.pkl'
# loss_file = '/home/cvip/faceswap-GAN/vegas_only_liveness.pkl'
# loss_file = './v8_finetune_scratch.pkl'
# loss_file = './v8_finetune.pkl'
# loss_file = './v7_with_frisbee.pkl'
# loss_file = '/home/cvip/faceswap-GAN/synthetic_faces.pkl'
loss_dict = pickle.load(open(loss_file, 'rb'))
plt.figure(1)
count = 1
iters = loss_dict['iters']
del loss_dict['iters']
for loss, values in loss_dict.items():
    plt.subplot(2, 4, count)
    plt.plot(iters, values)
    plt.xlabel(loss)
    count +=1
plt.show()
# z=1
#
# # recreate loss dict
# for i, iter in enumerate(loss_dict['iters']):
#     if iter > 39900:
#         last_index = i
#         break
# for key, list in loss_dict.items():
#     loss_dict[key] = loss_dict[key][:last_index]
#
# z=1