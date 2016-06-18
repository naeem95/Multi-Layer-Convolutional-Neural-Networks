load Data\mnist_uint8; 
train_x = double(reshape(train_x',28,28,60000))/255; 
test_x = double(reshape(test_x',28,28,10000))/255; 
train_y = double(train_y'); 
test_y = double(test_y'); 

% transpose to straighten the digits
for i=1:size(train_x,3) 
    train_x(:,:,i)=train_x(:,:,i)'; 
end 
for i=1:size(test_x,3) 
    test_x(:,:,i)=test_x(:,:,i)'; 
end 
%% network parameters 
% I -> 6C -> 6S -> 12C -> 12S -> 10F 
input_size=[size(train_x,1) size(train_x,2)]; 
output_size=size(train_y,1); 
num_slices=[6 12]; 
receptive_field_size=3; 
num_layers=6; 
error_function='cross_entropy';%'squared_error' 
 
%% optimization parameters 
alpha0=1e-3; 

%%load seed for same initialization of random numbers
load seed;
rng(seed);

%% setup the network 
layers=cell(1,num_layers); 
layers{1}.type='i'; 
layers{1}.num_slices=1; %input layer has only 1 slice (i.e., the input image) 
for l=2:2:num_layers-1 
    layers{l}.type='c'; 
    layers{l}.num_slices=num_slices(l/2); 
end 
num_subsmaplings=0; 
for l=3:2:num_layers-1 
    layers{l}.type='s'; 
    layers{l}.num_slices=layers{l-1}.num_slices;    %num of slices = num of slices in convolution layer 
    num_subsmaplings=num_subsmaplings+1; 
end 
layers{num_layers}.type='f'; 
layers{num_layers}.error_function=error_function; 
clear num_slices 

for k=1:num_layers 
     switch layers{k}.type
         case 'i'
            layers{k}.size = input_size;
         case 'c'
            layers{k}.size = layers{k-1}.size;
            layer_size = layers{k}.size;
            layers{k}.output = zeros((layer_size(1)),(layer_size(2)),layers{k}.num_slices);
         case 's'
            layers{k}.size = (layers{k-1}.size)/2;
            layer_size = layers{k}.size;
            layers{k}.output = zeros((layer_size(1)),(layer_size(2)),layers{k}.num_slices);
         case 'f'
            layers{k}.output = 10;
     end
end
 
% intialise weights in each layer 
for l=1:num_layers 
    switch layers{l}.type 
        case 'c' 
            %each slice 'looks at' all slices in previous layer 
            for s=1:layers{l}.num_slices 
                layers{l}.slice{s}.weights=randn(receptive_field_size,receptive_field_size,layers{l-1}.num_slices); 
                layers{l}.slice{s}.bias=randn; 
            end 
        case 'f' 
            sz=input_size;  %size of input, i.e., first layer 
            sz=sz/(2*num_subsmaplings); %size of last subsampled layer 
            % note that every output neuron is considered a slice to 
            % conform with the notation used in the rest of the layers 
            layers{l}.num_slices=output_size; 
            for k=1:layers{l}.num_slices 
                layers{l}.slice{k}.weights=randn(sz(1),sz(2),layers{l-1}.num_slices); 
                layers{l}.slice{k}.bias=randn;
            end 
    end 
end

for iter = 1:10
    for im = 1:size(train_x,3)
        for j=1:layers{2}.num_slices
            layers{2}.output(:,:,j) = Tanh(conv2(train_x(:,:,im),rot90(layers{2}.slice{j}.weights,2),'same') + layers{2}.slice{j}.bias);
            layers{2}.slice{j}.feature = layers{2}.output(:,:,j);
        end

        for l = 3:num_layers
            switch layers{l}.type
                case 's'
                    for j=1:layers{l}.num_slices
                        [layers{l}.output(:,:,j),maxInd] = max_pooling(layers{l-1}.output(:,:,j));
                        layers{l}.slice{j}.feature = layers{l}.output(:,:,j);
                        layers{l-1}.slice{j}.maxInd = maxInd;
                    end
                case 'c' 
                    for j=1:layers{l}.num_slices
                        sum_temp = zeros(size(layers{l-1}.output,1),size(layers{l-1}.output,2));
                        for k = 1:layers{l-1}.num_slices
                            sum_temp = sum_temp + conv2(layers{l-1}.output(:,:,k),rot90(layers{l}.slice{j}.weights(:,:,k),2),'same');
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
                        layers{l}.output = yk;
                        
                        if size(yk,1) == 1
                            yk = yk';
                        end
            end
        end

        % error at output layer
        delta_full = yk - train_y(:,im);               
        for j=1:10
            layers{l}.slice{j}.output = yk(j);
            layers{l}.slice{j}.delta = delta_full(j);
        end 

        % error at Pooling layer 2
        Del_P_2 = zeros(size(layers{5}.output));
        for cl=1:10
            Del_P_2 = Del_P_2 + layers{6}.slice{cl}.delta * layers{6}.slice{cl}.weights;
        end

        for j=1:layers{5}.num_slices
            layers{5}.slice{j}.delta = Del_P_2(:,:,j);
        end 

        % error at Convolution layer 2
        Del_C_2 = zeros(size(layers{4}.output));
        for fm=1:size(Del_C_2,3)
           Del_C_2(:,:,fm) = (1 - layers{4}.slice{fm}.feature.^2);
            for ih=1:size(Del_C_2,1)
                for iw=1:size(Del_C_2,2)
                    Del_C_2(ih,iw,fm) = Del_C_2(ih,iw,fm) * layers{5}.slice{fm}.delta(floor((ih+1)/2),floor((iw+1)/2)) * layers{4}.slice{fm}.maxInd(ih,iw);
                end
            end
            layers{4}.slice{fm}.delta = Del_C_2(:,:,fm);
        end

        % error at Pooling layer 1
        Del_P_1 = zeros(size(layers{3}.output));
        for cl=1:6
            for cl1=1:12
                Del_P_1(:,:,cl) = Del_P_1(:,:,cl) + conv2(layers{4}.slice{cl1}.delta,layers{4}.slice{cl1}.weights(:,:,cl),'same');
            end
            layers{3}.slice{cl}.delta = Del_P_1(:,:,cl);
        end

        % error at Convolutional layer 1
        Del_C_1 = zeros(size(layers{2}.output));
        for fm=1:size(Del_C_1,3)
           Del_C_1(:,:,fm) = (1 - layers{2}.slice{fm}.feature.^2);
            for ih=1:size(Del_C_1,1)
                for iw=1:size(Del_C_1,2)
                    Del_C_1(ih,iw,fm) = Del_C_1(ih,iw,fm) * layers{3}.slice{fm}.delta(floor((ih+1)/2),floor((iw+1)/2)) * layers{2}.slice{fm}.maxInd(ih,iw);
                end
            end
            layers{2}.slice{fm}.delta = Del_C_1(:,:,fm);
        end

        % Update bias at layer 6
        DB3 = delta_full; % gradient w.r.t bias                
        % Update weights at layer 3
        for cl=1:10
            layers{6}.slice{cl}.biasGrad = DB3(cl);
            layers{6}.slice{cl}.grad = layers{6}.slice{cl}.biasGrad * layers{5}.output; % gradient w.r.t weights
        end

        % Update weights and bias parameters at layer 4

        row = zeros(1,16);
        col = zeros(14,1);

        for fil2 = 1:12
            layers{4}.slice{fil2}.grad = zeros(3,3,6);
            for fil1 = 1:6
              padded_temp = [col layers{3}.slice{fil1}.feature col];
              padded_input = [row;padded_temp;row];
              layers{4}.slice{fil2}.grad(:,:,fil1) = layers{4}.slice{fil2}.grad(:,:,fil1) + conv2(padded_input,rot90(layers{4}.slice{fil2}.delta,2),'valid');
            end
            temp = layers{4}.slice{fil2}.delta;
            layers{4}.slice{fil2}.biasGrad = sum(temp(:));
        end

        % Update weights and bias parameters at layer 2                
        row = zeros(1,30);
        col = zeros(28,1);

        for fil2 = 1:6
            layers{2}.slice{fil2}.grad = zeros(3,3);
              padded_temp = [col train_x(:,:,im) col];
              padded_input = [row;padded_temp;row];
              layers{2}.slice{fil2}.grad(:,:) = layers{2}.slice{fil2}.grad(:,:) + conv2(padded_input,rot90(layers{2}.slice{fil2}.delta,2),'valid');
            temp = layers{2}.slice{fil2}.delta;
            layers{2}.slice{fil2}.biasGrad = sum(temp(:));
        end 

        for la=2:2:num_layers 
            for j = 1:layers{la}.num_slices
                layers{la}.slice{j}.weights = layers{la}.slice{j}.weights - (alpha0 * layers{la}.slice{j}.grad); 
                layers{la}.slice{j}.bias = layers{la}.slice{j}.bias - (alpha0 * layers{la}.slice{j}.biasGrad); 
            end
        end 
        im
        iter
    end
    accuracy(iter) = test_cnn(num_layers,layers,test_x,test_y);
    output = ['Accuracy = ',num2str(accuracy(iter)),' after Iteration ',num2str(iter)];
    disp(output);
end

