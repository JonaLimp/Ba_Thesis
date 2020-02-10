from utils import *
from data_deconv import *
import os

def VGG16_representation():

    dir_path = './convnet/Data/VGG16/'
    
    activation_save_path = os.path.join(dir_path, 'activation_dict_VGG.pickle')
    deconv_save_path = os.path.join(dir_path, 'deconv_dict_VGG.pickle')
    trans_rep_save_path = os.path.join(dir_path,'trans_rep_dict_VGG.pickle')
    csv_save_path = os.path.join(dir_path, 'csv_VGG.pickle')
    model = VGG16(weights='imagenet', include_top=True)
    layer_list = get_pool_conv_layer_list(get_layer_list(model))
    pdb.set_trace()
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    img_path = os.path.join(os.getcwd(), 'convnet/cat.jpg')
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    VGG_dict = get_top_k_activation(model, x, 1)

    max_elem = None
    max_value = 0.0


    df_VGG = pd.DataFrame(VGG_dict[0], columns=['img','layer','neuron','activation'])
    VGG_csv = df_VGG.to_csv()

    for idx,elem in enumerate(VGG_dict[0]):
        elem = list(elem)
        string = elem[1]
        elem[1] = string[:-2]
        VGG_dict[0][idx] = tuple(elem)



    VGG_deconv = deconvolve_data(x, VGG_dict, layer_list)


    
    pickle.dump(VGG_dict, open(activation_save_path, 'wb'))
    pickle.dump(VGG_deconv, open(deconv_save_path, 'wb'))
    pickle.dump(VGG_csv, open(csv_save_path, 'wb'))
    
    # trans_rep = translate_representations(deconv_save_path, trans_rep_save_path, layer_list, model, 50, 5)
    


    print('deconvolved images and csv are dumped')
    

def visualize_vgg_representations():

    dir_path = './convnet/Data/VGG16/'

    activation_save_path = os.path.join(dir_path, 'activation_dict_VGG.pickle')
    deconv_save_path = os.path.join(dir_path, 'deconv_dict_VGG.pickle')
    trans_rep_save_path = os.path.join(dir_path,'trans_rep_dict_VGG.pickle')
    csv_save_path = os.path.join(dir_path, 'csv_VGG.pickle')

    deconv = pickle.load(open(deconv_save_path,'rb'))
    csv =pickle.load(open(csv_save_path,'rb'))
    


    
    key = 'block5_conv3'
    pdb.set_trace()
    print(' ')
    # plt.imshow(deprocess_image(deconv[key][14][0][1]))

if __name__ == '__main__':


    VGG16_rep = False
    visualize_neuron = True

    if VGG16_rep ==True:
        VGG_rep_dict = VGG16_representation()

    if visualize_neuron == True:
        visualize_neurons()

