function [output] = convolution(input,filter)
output = zeros(size(input,1)-size(filter,1)+1);
for i = 1:size(input,1)-size(filter,1)+1
    for j = 1:size(input,2)-size(filter,2)+1
        output(i,j) = sum(sum(filter.*input(i:i+size(filter,1)-1,j:j+size(filter,2)-1)));
    end
end
