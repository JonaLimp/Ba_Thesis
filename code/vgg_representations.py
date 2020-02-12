from utils import *
from data_deconv import *
import pandas as pd
from keras.applications import imagenet_utils
import os


def VGG16_representation(img_name):

    dir_path = './convnet/Data/VGG16/{}/'.format(img_name[:-4])
    
    activation_save_path = os.path.join(dir_path, 'activation_dict_VGG.pickle')
    deconv_save_path = os.path.join(dir_path, 'deconv_dict_VGG.pickle')
    trans_rep_save_path = os.path.join(dir_path,'trans_rep_dict_VGG.pickle')
    csv_save_path = os.path.join(dir_path, 'csv_VGG.pickle')
    model = VGG16(weights='imagenet', include_top=True)
    layer_list = get_pool_conv_layer_list(get_layer_list(model))

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    pdb.set_trace()
    img_path = os.path.join(os.getcwd(), 'convnet/{}'.format(img_name))
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    VGG_dict = get_top_k_activation(model, x, 1)

    max_elem = None
    max_value = 0.0


    df_VGG = pd.DataFrame(VGG_dict[0], columns=['img','layer','neuron','activation'])
    VGG_csv = df_VGG.to_csv(csv_save_path)

    #just needed if layer has the wrong name
    # for idx,elem in enumerate(VGG_dict[0]):
    #     elem = list(elem)
    #     string = elem[1]
    #     elem[1] = string[:-2]
    #     VGG_dict[0][idx] = tuple(elem)



    VGG_deconv = deconvolve_data(x, VGG_dict, layer_list)


    
    pickle.dump(VGG_dict, open(activation_save_path, 'wb'))
    pickle.dump(VGG_deconv, open(deconv_save_path, 'wb'))
  
    
    # trans_rep = translate_representations(deconv_save_path, trans_rep_save_path, layer_list, model, 50, 5)
    


    print('deconvolved images and csv are dumped')
    

def visualize_vgg_representations(img_name):

    dir_path = './convnet/Data/VGG16/{}/'.format(img_name[:-4])
    img_path = os.path.join(dir_path,'representation_imgs/')
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    activation_save_path = os.path.join(dir_path, 'activation_dict_VGG.pickle')
    deconv_save_path = os.path.join(dir_path, 'deconv_dict_VGG.pickle')
    trans_rep_save_path = os.path.join(dir_path,'trans_rep_dict_VGG.pickle')
    csv_save_path = os.path.join(dir_path, 'csv_VGG.pickle')

    deconv = pickle.load(open(deconv_save_path,'rb'))
    # csv =pickle.load(open(csv_save_path,'rb'))
    pdb.set_trace()
    img_df = pd.read_csv(csv_save_path, names=['id','sample','layer','neuron','activation'])
    img_df.drop(labels='id',axis=1)

    elem_layer_col = img_df.layer.unique()
    
    pdb.set_trace()
    max_act_list = []

    plt.figure(figsize=(100 ,100 ))

    for idx,layer in enumerate(elem_layer_col[1:]):
        layer_df = img_df[img_df['layer'] == layer]
        layer_df['activation'] = pd.to_numeric(layer_df['activation'])
        layer_df.sort_values(by='activation', inplace=True,ascending=False)
        max_act_list.append(layer_df.iloc[0].as_matrix())


        # plt.subplot(4,4,idx+1)
        layer_name = max_act_list[idx][2]
        neuron_idx = int(max_act_list[idx][3])
        # plt.imshow(postprocess(deconv[layer_name][neuron_idx][0][1]))
        plt.imshow(deprocess_image(deconv[layer_name][neuron_idx][0][1]))

        plt.savefig(os.path.join(img_path, f'deconv_img_{layer}'))

    # plt.show()
    pdb.set_trace()
    # plt.imshow(deprocess_image(deconv[key][14][0][1]))

if __name__ == '__main__':

    img_name = 'cat.jpg'

    VGG16_rep = True
    visualize_vgg = True

    if VGG16_rep ==True:
        VGG_rep_dict = VGG16_representation(img_name)

    if visualize_vgg == True:
        visualize_vgg_representations(img_name)

