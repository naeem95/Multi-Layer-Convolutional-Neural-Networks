function [accuracy] = test_cnn(num_layers,layers,test_x,test_y)
miss = 0;
size_test=size(test_x,3);
for im=1:size_test
    for j=1:layers{2}.num_slices        
        layers{2}.output(:,:,j) = Tanh(conv2(test_x(:,:,im),rot90(layers{2}.slice{j}.weights,2),'same') + layers{2}.slice{j}.bias);
        layers{2}.slice{j}.feature = layers{2}.output(:,:,j);
    end

    for l = 3:num_layers
        switch layers{l}.type
            case 's'
                for j=1:layers{l}.num_slices
                    [layers{l}.output(:,:,j),maxInd] = max_pooling(layers{l-1}.output(:,:,j));
                    layers{l}.slice{j}.feature = layers{l}.output(:,:,j);
                     layers{l}.output(:,:,j) = layers{l}.slice{j}.feature;
                    layers{l-1}.slice{j}.maxInd = maxInd;
                end
            case 'c' 
                for j=1:layers{l}.num_slices
                    tempMat = zeros(size(layers{l-1}.output,1),size(layers{l-1}.output,2),(layers{l-1}.num_slices));
                    for k = 1:layers{l-1}.num_slices
                        tempMat(:,:,k) = conv2(layers{l-1}.output(:,:,k),rot90(layers{l}.slice{j}.weights(:,:,k),2),'same');
                    end
                    sum_temp = zeros(size(tempMat,1),size(tempMat,2));
                    for k = 1:layers{l-1}.num_slices
                        sum_temp = sum_temp + tempMat(:,:,k);
                    end
                    layers{l}.output(:,:,j) = Tanh(sum_temp + layers{l}.slice{j}.bias);
                    layers{l}.slice{j}.feature = layers{l}.output(:,:,j);
                end
            case 'f' 
                    for j=1:10
                        layers{l}.output(j) = dot(reshape(layers{l-1}.output, 1, []),(reshape(layers{l}.slice{j}.weights, 1, []))') + layers{l}.slice{j}.bias;
                    end

                    for j=1:10
                        layers{l}.slice{j}.activation = layers{l}.output(j);
                    end

                    yk = (SoftMax(layers{l}.output))';
        end
    end
    
    [train_m,train_I] = max(yk);
    [test_m,test_I] = max(test_y(:,im));
    if train_I ~= test_I
        miss = miss + 1;
    end
end
accuracy = ((size_test - miss)/ size_test) * 100;
disp(accuracy);

end