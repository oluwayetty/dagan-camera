import scipy.misc
import imageio
import numpy as np

def unstack(np_array):
    new_list = []
    for i in range(np_array.shape[0]):
        temp_list = np_array[i]
        new_list.append(temp_list)
    return new_list

def sample_generator(num_generations, sess, same_images, inputs, dropout_rate, dropout_rate_value, data, batch_size,
                     file_name, input_a, training_phase, z_input, z_vectors, epoch):


    input_images, generated = sess.run(same_images, feed_dict={ input_a: inputs,
                                                                dropout_rate: dropout_rate_value,
                                                                training_phase: False,
                                                                z_input: batch_size*[z_vectors[0]]})

    input_images_list = np.zeros(shape=(batch_size, num_generations, input_images.shape[-3], input_images.shape[-2],
                                        input_images.shape[-1]))
    generated_list = np.zeros(shape=(batch_size, num_generations, generated.shape[-3], generated.shape[-2],
                                     generated.shape[-1]))
    for i in range(num_generations):
        input_images, generated = sess.run(same_images, feed_dict={z_input: batch_size*[z_vectors[i]],
                                                                   input_a: inputs,
                                                                   training_phase: False,
                                                                   dropout_rate: dropout_rate_value})

        input_images_reconstruct, generated_reconstruct = data.reconstruct_original(input_images), data.reconstruct_original(generated)

        assert (input_images_reconstruct.shape) == (generated_reconstruct.shape)
        for j in range(len(input_images_reconstruct)):
            print("Saving input and generated images at index {}".format(i))

            imageio.imwrite(file_name.split(".png")[0] + "_input_{}_{}.png".format(i,j), input_images_reconstruct[j])
            imageio.imwrite(file_name.split(".png")[0] + "_generated_{}_{}.png".format(i,j), generated_reconstruct[j])


def sample_two_dimensions_generator(sess, same_images, inputs,
                                    dropout_rate, dropout_rate_value, data,
                                    batch_size, file_name, input_a,
                                    training_phase, z_input, z_vectors, epoch):
    num_generations = z_vectors.shape[0]
    row_num_generations = int(np.sqrt(num_generations))
    column_num_generations = int(np.sqrt(num_generations))

    input_images, generated = sess.run(same_images, feed_dict={input_a: inputs, dropout_rate: dropout_rate_value,
                                                                  training_phase: False,
                                                                  z_input: batch_size*[z_vectors[0]]})

    input_images_list = np.zeros(shape=(batch_size, num_generations, input_images.shape[-3], input_images.shape[-2],
                                        input_images.shape[-1]))
    generated_list = np.zeros(shape=(batch_size, num_generations, generated.shape[-3], generated.shape[-2],
                                     generated.shape[-1]))

    for i in range(num_generations):
        input_images, generated = sess.run(same_images, feed_dict={z_input: batch_size*[z_vectors[i]],
                                                                      input_a: inputs,
                                                                      training_phase: False, dropout_rate:
                                                                      dropout_rate_value})

        input_images_reconstruct2D, generated_reconstruct2D = data.reconstruct_original(input_images), data.reconstruct_original(generated)

        assert (input_images_reconstruct2D.shape) == (generated_reconstruct2D.shape)
        for j in range(len(generated_reconstruct2D)):
            print("Saving input2D and generated2D images at index {}".format(i))
            imageio.imwrite(file_name.split(".png")[0] + "_input2D_{}_{}.png".format(i,j), input_images_reconstruct2D[j])
            imageio.imwrite(file_name.split(".png")[0] + "_generated2D_{}_{}.png".format(i,j), generated_reconstruct2D[j])
