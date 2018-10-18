import cv2
import os
import numpy as np
import random
import pickle
from skimage.draw import circle
from skimage.draw import rectangle
from skimage.draw import polygon
from pathlib import Path
home = str(Path.home())

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='parser')
	parser.add_argument('--train-size', type=int, default=9800)
	parser.add_argument('--test-size', type=int, default=200)
	parser.add_argument('--image-size', type=int, default=75)
	parser.add_argument('--size', type=int, default=5)
	parser.add_argument('--closest', type=int, default=3)
	args = parser.parse_args()
	config = '_'.join(map(str, [args.train_size, args.test_size, args.image_size, args.size, args.closest]))
	train_size = args.train_size
	test_size = args.test_size
	img_size = args.image_size
	size = args.size

slack = 5
num_shape = 2
num_rel_qst = 5
question_size = 11 + (num_rel_qst - 3)  ##6 for one-hot vector of color, 2 for question type, 3 for question subtype
"""Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""
answer_size_before_color = 10 +num_shape  # 0 ~ 9 answer_dict
answer_size_before_count = 4  # 0 ~ 4

# 3 for question subtype
"""Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""

nb_questions = 10



colors = [
    (255, 0, 0),  ##r
    (0, 255, 0),  ##g
    (0, 0, 255),  ##b
    (255, 156, 0),  ##orange
    (148, 0, 211),  ##darkviolet
    (255, 255, 0)  ##y
]


color_dict = {
        0: 'r',
        1: 'g',
        2: 'b',
        3: 'o',
        4: 'v',
        5: 'y',
    }


question_type_dict = {
            0: 's',
            1: 'h',
            2: 'v',
            3: 'cl_c',
            4: 'f_c',
            5: 'co',
            6: 'cl_s',
            7: 'f_s'
        }



# answer_dict = {
#             0: 'y',
#             1: 'n',
#             2: 'rec',
#             3: 'cir',
#             4: 'tri',
#             5: '1',
#             6: '2',
#             7: '3',
#             8: '4',
#             9: '5',
#             10: '6',
#             11: 'r',
#             12: 'g',
#             13: 'b',
#             14: 'o',
#             15: 'v',
#             16: 'y'
#         }

answer_dict = {
            0: 'y',
            1: 'n',
            2: 'rec',
            3: 'cir',
            4: '1',
            5: '2',
            6: '3',
            7: '4',
            8: '5',
            9: '6',
            10: 'r',
            11: 'g',
            12: 'b',
            13: 'o',
            14: 'v',
            15: 'y'
        }




def draw_triangle(img, img_size, x, y, size, color):
    # img = np.zeros((img_size, img_size, 3))
    size = size * 1.5
    upper = np.array((0, size))
    spin_mat = np.array(
        [np.cos(np.pi / 3 * 2), -np.sin(np.pi / 3 * 2), np.sin(np.pi / 3 * 2),
         np.cos(np.pi / 3 * 2)]).reshape(2, 2)
    lower_right = np.matmul(spin_mat, upper)
    lower_left = np.matmul(spin_mat, lower_right)

    vertices = np.stack((upper, lower_right, lower_left), axis=1).reshape(-1, 3)
    vertices[0, :] = vertices[0, :] + x
    vertices[1, :] = vertices[1, :] + y

    tmp = vertices[0, :].copy()
    vertices[0, :] = vertices[1, :]
    vertices[1, :] = tmp

    rr, cc = polygon(vertices[0, :], vertices[1, :])

    img[rr, cc, 0] = color[0]
    img[rr, cc, 1] = color[1]
    img[rr, cc, 2] = color[2]

    #     print(rr)
    #     print(cc)
    return img



def center_generate(objects):
    while True:
        pas = True

        center = np.random.randint(0 + size + slack, img_size - size - slack, 2)
        if len(objects) > 0:
            for name, c, shape in objects:
                if ((center - c) ** 2).sum() < (3 * (size * 2) ** 2):
                    pas = False
        if pas:
            return center


def build_dataset():
    objects = []
    img = np.ones((img_size, img_size, 3)) * 255
    for color_id, color in enumerate(colors):
        center = center_generate(objects)
        shape = np.random.randint(num_shape)
        if shape == 0:
            start = (center[0] - size, center[1] - size)
            end = (center[0] + size, center[1] + size)
            # cv2.rectangle(img, start, end, color, -1)
            rr, cc = rectangle(start, end)
            img[rr, cc] = color
            objects.append((color_id, center, 'rec'))

        elif shape == 1:
            center_ = (center[0], center[1])
            # cv2.circle(img, center_, size, color, -1)
            rr, cc = circle(*center_, size + 1)
            img[rr, cc] = color

            objects.append((color_id, center, 'cir'))

        elif shape == 2:
            center_ = (center[1] , center[0])
            img = draw_triangle(img, img_size, *center_, size  , color)
            objects.append((color_id, center, 'tri'))



    rel_questions = []
    norel_questions = []
    rel_answers = []
    norel_answers = []
    # """Non-relational questions"""
    for _ in range(nb_questions):
        question = np.zeros((question_size))
        color = random.randint(0, 5)
        question[color] = 1
        question[6] = 1
        subtype = random.randint(0, 2)
        question[subtype + 8] = 1
        norel_questions.append(question)
        """Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""
        if subtype == 0:
            """query shape->rectangle/circle"""
            if objects[color][2] == 'rec':
                answer = 2
            elif objects[color][2] == 'cir':
                answer = 3
            elif objects[color][2] == 'tri':
                answer = 4
            else:
                print('error in dat')
                exit()

        elif subtype == 1:
            """query horizontal position->yes/no"""
            if objects[color][1][0] < img_size / 2:
                answer = 0
            else:
                answer = 1

        elif subtype == 2:
            """query vertical position->yes/no"""
            if objects[color][1][1] < img_size / 2:
                answer = 0
            else:
                answer = 1
        norel_answers.append(answer)

    """Relational questions"""
    for i in range(nb_questions):
        question = np.zeros((question_size))
        color = random.randint(0, 5)
        question[color] = 1
        question[7] = 1
        subtype = random.randint(0, 2)
        question[subtype + 8] = 1
        rel_questions.append(question)

        if subtype == 0:
            """closest-to->rectangle/circle"""
            my_obj = objects[color][1]
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            dist_list[dist_list.index(0)] = (img_size ** 2) * 2 #max distance
            closest = dist_list.index(min(dist_list))
            if objects[closest][2] == 'rec':
                answer = 2
            elif objects[closest][2] == 'cir':
                answer = 3
            elif objects[closest][2] == 'tri':
                answer = 4
            else:
                print('error in data')
                exit()
            # answer = objects[closest][0] + answer_size_before_color

        elif subtype == 1:
            """furthest-from->rectangle/circle"""
            my_obj = objects[color][1]
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            furthest = dist_list.index(max(dist_list))

            if objects[furthest][2] == 'rec':
                answer = 2
            elif objects[furthest][2] == 'cir':
                answer = 3
            elif objects[furthest][2] == 'tri':
                answer = 4
            else:
                print('error in data')
                exit()

            # answer = objects[furthest][0] + answer_size_before_color

        elif subtype == 2:
            """count->1~6"""
            my_obj = objects[color][2]
            count = -1
            for obj in objects:
                if obj[2] == my_obj:
                    count += 1

            answer = count + answer_size_before_count

        rel_answers.append(answer)

    relations = (rel_questions, rel_answers)
    norelations = (norel_questions, norel_answers)

    # img = img / 255.
    dataset = (img, relations, norelations)
    return dataset

def build_dataset_all_question():
    objects = []
    img = np.ones((img_size, img_size, 3)) * 255
    for color_id, color in enumerate(colors):
        center = center_generate(objects)
        shape = np.random.randint(num_shape)
        if shape == 0:
            start = (center[0] - size, center[1] - size)
            end = (center[0] + size, center[1] + size)
            # cv2.rectangle(img, start, end, color, -1)
            rr, cc = rectangle(start, end)
            img[rr, cc] = color
            objects.append((color_id, center, 'rec'))

        elif shape == 1:
            center_ = (center[0], center[1])
            # cv2.circle(img, center_, size, color, -1)
            rr, cc = circle(*center_, size + 1)
            img[rr, cc] = color

            objects.append((color_id, center, 'cir'))

        # elif shape == 2:
        #     center_ = (center[1] , center[0])
        #     img = draw_triangle(img, img_size, *center_, size  , color)
        #     objects.append((color_id, center, 'tri'))



    rel_questions = []
    norel_questions = []
    rel_answers = []
    norel_answers = []
    # """Non-relational questions"""
    for color in color_dict.keys():
        for subtype in [0, 1, 2]:
            question = np.zeros((question_size))
            # color = random.randint(0, 5)
            question[color] = 1
            question[6] = 1
            # subtype = random.randint(0, 2)
            question[subtype + 8] = 1
            norel_questions.append(question)
            """Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""
            if subtype == 0:
                """query shape->rectangle/circle"""
                if objects[color][2] == 'rec':
                    answer = 2
                elif objects[color][2] == 'cir':
                    answer = 3
                # elif objects[color][2] == 'tri':
                #     answer = 4
                else:
                    print('error in dat')
                    exit()

            elif subtype == 1:
                """query horizontal position->yes/no"""
                if objects[color][1][0] < img_size / 2:
                    answer = 0
                else:
                    answer = 1

            elif subtype == 2:
                """query vertical position->yes/no"""
                if objects[color][1][1] < img_size / 2:
                    answer = 0
                else:
                    answer = 1
            norel_answers.append(answer)

    """Relational questions"""
    for color in color_dict.keys():
        for subtype in range(num_rel_qst):
            question = np.zeros((question_size))
            # color = random.randint(0, 5)
            question[color] = 1
            question[7] = 1
            # subtype = random.randint(0, 2)
            question[subtype + 8] = 1
            rel_questions.append(question)

            if subtype == 0:
                """closest-to->rectangle/circle"""
                my_obj = objects[color][1]
                dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
                dist_list[dist_list.index(0)] = (img_size ** 2) * 2 #max distance
                closest = dist_list.index(min(dist_list))


                answer = objects[closest][0] + answer_size_before_color

            elif subtype == 1:
                """furthest-from->rectangle/circle"""
                my_obj = objects[color][1]
                dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
                furthest = dist_list.index(max(dist_list))



                answer = objects[furthest][0] + answer_size_before_color

            elif subtype == 2:
                """count->1~6"""
                my_obj = objects[color][2]
                count = -1
                for obj in objects:
                    if obj[2] == my_obj:
                        count += 1

                answer = count + answer_size_before_count


            elif subtype == 3:
                """closest-to->rectangle/circle"""
                my_obj = objects[color][1]
                dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
                dist_list[dist_list.index(0)] = (img_size ** 2) * 2  # max distance
                closest = dist_list.index(min(dist_list))

                if objects[closest][2] == 'rec':
                    answer = 2
                elif objects[closest][2] == 'cir':
                    answer = 3
                # elif objects[closest][2] == 'tri':
                #     answer = 4
                else:
                    print('error in data')
                    exit()


            elif subtype == 4:
                """furthest-from->rectangle/circle"""
                my_obj = objects[color][1]
                dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
                furthest = dist_list.index(max(dist_list))

                if objects[furthest][2] == 'rec':
                    answer = 2
                elif objects[furthest][2] == 'cir':
                    answer = 3
                # elif objects[furthest][2] == 'tri':
                #     answer = 4
                else:
                    print('error in data')
                    exit()





            rel_answers.append(answer)

    relations = (rel_questions, rel_answers)
    norelations = (norel_questions, norel_answers)

    # img = img / 255.
    dataset = (img, relations, norelations)
    return dataset


def generate_data(data_option=None):
	if data_option:
		dirs = home + '/data/sortofclevr3/{}'.format(data_option)
	else:
		dirs = home + '/data/sortofclevr3'

	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))

	filename = os.path.join(dirs, 'sort-of-clevr3-train.pickle')

	if not os.path.exists(filename):

		print('building test datasets...')
		test_datasets = [build_dataset() for _ in range(test_size)]
		print('building train datasets...')
		train_datasets = [build_dataset() for _ in range(train_size)]

		# img_count = 0
		# cv2.imwrite(os.path.join(dirs,'{}.png'.format(img_count)), cv2.resize(train_datasets[0][0]*255, (512,512)))

		print('saving datasets...')

		with  open(os.path.join(dirs, 'sort-of-clevr3-train.pickle'), 'wb') as f:
			pickle.dump(train_datasets, f)
		with  open(os.path.join(dirs, 'sort-of-clevr3-val.pickle'), 'wb') as f:
			pickle.dump(test_datasets, f)
		print('datasets saved at {}'.format(dirs))

if __name__ == '__main__':
	generate_data(config)
