% Width: 640, Height: 480

count_offset = 0;

%% Superimpose fruit onto arena picture 
mnval = 0.2;
mxval = 0.5;
offset = 30;    % Diagonaly offset fruit from corner of grid
% Left most corner of grids
grid_point = [[0 0]; [213 0]; [426 0]; [0 160]; [213 160]; [426 160]; [0 320]; [213 320]; [426 320]];
% Get left most corner of fruit image
fruit_point1 = [];

imagefiles = dir('Pictures/Fruits/*.png');
nfiles = length(imagefiles);    % Number of files found

count = 0;

%% Read all images into images{i}
for i=1:1:nfiles
    currentfilename = imagefiles(i).name;
    currentimage = imread("Pictures/Fruits/" + currentfilename);
    images{i} = currentimage;
end

%% Segregate into double fruits
% Capsicum and green apple
capsicum_gApple = images(1: 24);
% Orange and red apple
rApple_Orange = images(37: 60);
% green apple and orange
orange_gApple = images([13: 24, 37:48]);

%% Superimpose two fruits
images = capsicum_gApple;
for i=1:12
    i2 = i+12;
    % Scale picture first randomly
    scale1 = mnval + rand*(mxval-mnval);
    scale2 = mnval + rand*(mxval-mnval);
    current_scaled_image1 = imresize(images{i}, scale1);
    current_scaled_image2 = imresize(images{i2}, scale2);

    % Find left most corner of fruit
    index1 = current_scaled_image1 ~= 0;
    index_fruit1 = index1(:, :, 1) + index1(:, :, 2) + index1(:, :, 3);

    index2 = current_scaled_image2 ~= 0;
    index_fruit2 = index2(:, :, 1) + index2(:, :, 2) + index2(:, :, 3);

    % First fruit
    max_y1 = 0;
    min_x1 = 0;
    for j = 1:480
        row = index_fruit1(j, :);
        if (max(row) ~= 0)  % Top most pixel detected
            max_y1 = j;
            break
        end
    end

    for j = 1:640
        col = index_fruit1(:, j);
        if (max(col) ~= 0)  % left most pixel detected
            min_x1 = j;
            break
        end
    end

    % Second fruit
    max_y2 = 0;
    min_x2 = 0;
    for j = 1:480
        row = index_fruit2(j, :);
        if (max(row) ~= 0)  % Top most pixel detected
            max_y2 = j;
            break
        end
    end

    for j = 1:640
        col = index_fruit2(:, j);
        if (max(col) ~= 0)  % left most pixel detected
            min_x2 = j;
            break
        end
    end

    if i == 1 | i2 == 13
        fruit_point1 = [max_y1 min_x1];
        fruit_point2 = [max_y2 min_x2];
    else
        fruit_point1 = [fruit_point1; [max_y1 min_x1]];
        fruit_point2 = [fruit_point2; [max_y2 min_x2]];
    end

    current_fruit_point_x1 = fruit_point1(i, 2);
    current_fruit_point_y1 = fruit_point1(i, 1);

    current_fruit_point_x2 = fruit_point2(i, 2);
    current_fruit_point_y2 = fruit_point2(i, 1);

    % Remove left and top padding
    current_scaled_image1 = current_scaled_image1(current_fruit_point_y1 - 1: end, current_fruit_point_x1 - 1: end, :);
    index_fruit1 = index_fruit1(current_fruit_point_y1 - 1: end, current_fruit_point_x1 - 1: end, :);

    current_scaled_image2 = current_scaled_image2(current_fruit_point_y2 - 1: end, current_fruit_point_x2 - 1: end, :);
    index_fruit2 = index_fruit2(current_fruit_point_y2 - 1: end, current_fruit_point_x2 - 1: end, :);
    % Superimpose
    % Each fruit:
    % For each background, superimpose the fruit on each 9 grid
    % 1. Loop through backgrounds
    % 2. For each background loop, loop through 9 grid points
    % 3. Save picture
    backgroundImage = dir('Pictures/Background/*.png');
    nfiles_bg = length(backgroundImage);    % Number of files found

    for k=1:nfiles_bg
        currentfilename_bg = backgroundImage(k).name;
        currentimage_bg = imread("Pictures/Background/" + currentfilename_bg);
        images_bg{k} = currentimage_bg;

        % Loop through grid
        for j = 1: 9
            start_x_point = grid_point(j, 1) + offset;
            start_y_point = grid_point(j, 2) + offset;

            superimposed_image = currentimage_bg;
            size2 = size(currentimage_bg);
            labelled_image = zeros(size2(1), size2(2), 'uint8');

            % Get distance from starting point to edge of picture
            x_dist = 640 - start_x_point;
            y_dist = 480 - start_y_point;

            size_image_x = size(current_scaled_image1);
            size_image_x = size_image_x(2);
            size_image_y = size(current_scaled_image1);
            size_image_y = size_image_y(1);

            x_diff = x_dist - size_image_x;
            y_diff = y_dist - size_image_y;

            x_bound = start_x_point + size_image_x;
            y_bound = start_y_point + size_image_y;

            if x_diff > 0
                range_x = x_bound;
            else
                range_x = 640;
            end

            if y_diff > 0
                range_y = y_bound;
            else
                range_y = 480;
            end


            x_index = 1;
            y_index = 1;
            for x_bg = start_x_point: range_x - 1
                for y_bg = start_y_point: range_y - 1
                    x_fruit = x_index;
                    y_fruit = y_index;

                    if index_fruit1(y_fruit, x_fruit) ~= 0
                        superimposed_image(y_bg, x_bg, :) = current_scaled_image1(y_fruit, x_fruit, :);
                        
                        % Capsicum
                        labelled_image(y_bg, x_bg, :) = 5;

                    end

                    y_index = y_index + 1;
                end
                y_index = 1;
                x_index = x_index + 1;
            end

            %% Second fruit
            start_x_point = grid_point(10-j, 1) + offset;
            start_y_point = grid_point(10-j, 2) + offset;

            % Get distance from starting point to edge of picture
            x_dist = 640 - start_x_point;
            y_dist = 480 - start_y_point;

            size_image_x = size(current_scaled_image2);
            size_image_x = size_image_x(2);
            size_image_y = size(current_scaled_image2);
            size_image_y = size_image_y(1);

            x_diff = x_dist - size_image_x;
            y_diff = y_dist - size_image_y;

            x_bound = start_x_point + size_image_x;
            y_bound = start_y_point + size_image_y;

            if x_diff > 0
                range_x = x_bound;
            else
                range_x = 640;
            end

            if y_diff > 0
                range_y = y_bound;
            else
                range_y = 480;
            end


            x_index = 1;
            y_index = 1;
            for x_bg = start_x_point: range_x - 1
                for y_bg = start_y_point: range_y - 1
                    x_fruit = x_index;
                    y_fruit = y_index;

                    if index_fruit2(y_fruit, x_fruit) ~= 0
                        superimposed_image(y_bg, x_bg, :) = current_scaled_image2(y_fruit, x_fruit, :);
                        
                        % Green apple
                        labelled_image(y_bg, x_bg, :) = 2;

                    end

                    y_index = y_index + 1;
                end
                y_index = 1;
                x_index = x_index + 1;
            end
            
            number = count + count_offset;

            imwrite(labelled_image, "Pictures/labeled/image_" + number + "_label.png");
            imwrite(superimposed_image, "Pictures/Superimposed/image_" + number + ".png");
            count = count + 1;
        end
    end
end
  

