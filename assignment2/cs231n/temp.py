patches = self.generate_patches(name_list[i])
 -
 -            count_i = np.zeros(n_classes)
 -            for patch_i in patches:
 -                # print(patch_i.shape)
 -                # subtract the mean of train set, be careful with the index [0] as it is required, since the output has first dimension of N, where N=1
 -                pred_i = model.predict(np.subtract(patch_i, model_mean)[None, :, :, :])[0]
 -                pred_each_i = [pred_i[:,:,k] for k in range(n_classes)]
 -                count_each_i = np.array([np.sum(p)/self.scaling for p in pred_each_i])
 -                count_each_i[count_each_i < 0] = 0 # if predicted count is negative then we treat it as zero
 -
 -                # print('\n>>>>', count_each_i, '<<<<\n')
 -                if mode == 'round_first':
 -                    count_i += np.round(count_each_i)
 -                elif mode == 'round_after':
 -                    count_i += count_each_i
 -
 -            if mode == 'round_first':
 -                result[i] = count_i
 -            elif mode == 'round_after':
 -                result[i] = np.round(count_i)
 -
 -            loop_count += 1
 -            gc.collect()
 -            bar.update(loop_count)