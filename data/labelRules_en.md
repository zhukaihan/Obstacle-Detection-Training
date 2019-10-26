# How to Take Pictures: 
* You need to stand on the ground (not on the grass, not on the table). 
* The obstacle needs to be on the ground (not on the table, they don't walk on tables). No pictures of a table full of Doritos. 
* Since we are training a Fully Convolutional Neural Network, the image can be of any size. However, whey you are taking pictures with the app, do it horizontally. 
* We can take pictures of an obstacle from different angles, lightings, distance, etc. It helps fulfilling the requirements, also helps to make our model to be more robust. 

# How to Label: 
* The box needs to be bounding the obstacle TIGHTLY. In other words, each of the four sides of the bounding box needs to have contact with the obstacle at some point. 
* Label everything in 7 meters. If it is outdoor, label obvious obstacles. 
* Definition of different labels: 
  * Obstacle - An obstacle. 
    * We need different kind of obstacles, large or small. For example, tables, different tables, tables with different colors, tables with different sizes, tables with different legs, a full table, half of a table, etc. 
    * Wall is an obstacle as well. 
  * Edge - Edge of the sidewalk, where there is a bump and people will tripped over or fall down. 
    * A continuous edge will be labeled in one box. If the edge is a straight edge, this edge should be the diagnal line of its bounding box. 
    * Don't label an edge with 10 boxes. Just one, even if it is very very big. 
  * Pothole - A hole on the ground. Usually formed by damaging the ground. 
    * La Jolla has a pretty well built road. 
  * Uplift - A part of the sidewalk where is an uplift... The sidewalk is consist of concretes. On tile of concrete against another to form the sidewalk. There is also trees planted next to the sidewalk. As the tree grows, its root goes under the concrete. As its root thickens, it pops up the concrete. Then, one tile of concrete is unlevel with the other. The part where they are unlevel is called uplift. 

# Enphasize on the Importance of Labeling: 
We have a little over 300 images. Traditionally, we need at least 1000 images to learn a new class. Therefore, we cannot afford to label images incorrectly. 
