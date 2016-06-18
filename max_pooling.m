function [output,maxInd] = max_pooling(input)
    output = zeros(size(input)/2);
    maxInd = zeros(size(input));
    for r=1:2:size(input,1)
        for c=1:2:size(input,2)
            A = [input(r,c) input(r+1,c) input(r,c+1) input(r+1,c+1)];
            [M,I] = max(A);
            switch I
                case 1
                    maxInd(r,c) = 1;
                case 2
                    maxInd(r+1,c) = 1;
                case 3
                    maxInd(r,c+1) = 1;
                case 4
                    maxInd(r+1,c+1) = 1;
            end
            output((r+1)/2,(c+1)/2) = M;
        end
    end
end