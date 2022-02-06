face_mesh_lst = [(127, 34),  (34, 139),  (139, 127), (11, 0),    (0, 37),    (37, 11),
    (232, 231), (231, 120), (120, 232), (72, 37),   (37, 39),   (39, 72),
    (128, 121), (121, 47),  (47, 128),  (232, 121), (121, 128), (128, 232),
    (104, 69),  (69, 67),   (67, 104),  (175, 171), (171, 148), (148, 175),
    (118, 50),  (50, 101),  (101, 118), (73, 39),   (39, 40),   (40, 73),
    (9, 151),   (151, 108), (108, 9),   (48, 115),  (115, 131), (131, 48),
    (194, 204), (204, 211), (211, 194), (74, 40),   (40, 185),  (185, 74),
    (80, 42),   (42, 183),  (183, 80),  (40, 92),   (92, 186),  (186, 40),
    (230, 229), (229, 118), (118, 230), (202, 212), (212, 214), (214, 202),
    (83, 18),   (18, 17),   (17, 83),   (76, 61),   (61, 146),  (146, 76),
    (160, 29),  (29, 30),   (30, 160),  (56, 157),  (157, 173), (173, 56),
    (106, 204), (204, 194), (194, 106), (135, 214), (214, 192), (192, 135),
    (203, 165), (165, 98),  (98, 203),  (21, 71),   (71, 68),   (68, 21),
    (51, 45),   (45, 4),    (4, 51),    (144, 24),  (24, 23),   (23, 144),
    (77, 146),  (146, 91),  (91, 77),   (205, 50),  (50, 187),  (187, 205),
    (201, 200), (200, 18),  (18, 201),  (91, 106),  (106, 182), (182, 91),
    (90, 91),   (91, 181),  (181, 90),  (85, 84),   (84, 17),   (17, 85),
    (206, 203), (203, 36),  (36, 206),  (148, 171), (171, 140), (140, 148),
    (92, 40),   (40, 39),   (39, 92),   (193, 189), (189, 244), (244, 193),
    (159, 158), (158, 28),  (28, 159),  (247, 246), (246, 161), (161, 247),
    (236, 3),   (3, 196),   (196, 236), (54, 68),   (68, 104),  (104, 54),
    (193, 168), (168, 8),   (8, 193),   (117, 228), (228, 31),  (31, 117),
    (189, 193), (193, 55),  (55, 189),  (98, 97),   (97, 99),   (99, 98),
    (126, 47),  (47, 100),  (100, 126), (166, 79),  (79, 218),  (218, 166),
    (155, 154), (154, 26),  (26, 155),  (209, 49),  (49, 131),  (131, 209),
    (135, 136), (136, 150), (150, 135), (47, 126),  (126, 217), (217, 47),
    (223, 52),  (52, 53),   (53, 223),  (45, 51),   (51, 134),  (134, 45),
    (211, 170), (170, 140), (140, 211), (67, 69),   (69, 108),  (108, 67),
    (43, 106),  (106, 91),  (91, 43),   (230, 119), (119, 120), (120, 230),
    (226, 130), (130, 247), (247, 226), (63, 53),   (53, 52),   (52, 63),
    (238, 20),  (20, 242),  (242, 238), (46, 70),   (70, 156),  (156, 46),
    (78, 62),   (62, 96),   (96, 78),   (46, 53),   (53, 63),   (63, 46),
    (143, 34),  (34, 227),  (227, 143), (123, 117), (117, 111), (111, 123),
    (44, 125),  (125, 19),  (19, 44),   (236, 134), (134, 51),  (51, 236),
    (216, 206), (206, 205), (205, 216), (154, 153), (153, 22),  (22, 154),
    (39, 37),   (37, 167),  (167, 39),  (200, 201), (201, 208), (208, 200),
    (36, 142),  (142, 100), (100, 36),  (57, 212),  (212, 202), (202, 57),
    (20, 60),   (60, 99),   (99, 20),   (28, 158),  (158, 157), (157, 28),
    (35, 226),  (226, 113), (113, 35),  (160, 159), (159, 27),  (27, 160),
    (204, 202), (202, 210), (210, 204), (113, 225), (225, 46),  (46, 113),
    (43, 202),  (202, 204), (204, 43),  (62, 76),   (76, 77),   (77, 62),
    (137, 123), (123, 116), (116, 137), (41, 38),   (38, 72),   (72, 41),
    (203, 129), (129, 142), (142, 203), (64, 98),   (98, 240),  (240, 64),
    (49, 102),  (102, 64),  (64, 49),   (41, 73),   (73, 74),   (74, 41),
    (212, 216), (216, 207), (207, 212), (42, 74),   (74, 184),  (184, 42),
    (169, 170), (170, 211), (211, 169), (170, 149), (149, 176), (176, 170),
    (105, 66),  (66, 69),   (69, 105),  (122, 6),   (6, 168),   (168, 122),
    (123, 147), (147, 187), (187, 123), (96, 77),   (77, 90),   (90, 96),
    (65, 55),   (55, 107),  (107, 65),  (89, 90),   (90, 180),  (180, 89),
    (101, 100), (100, 120), (120, 101), (63, 105),  (105, 104), (104, 63),
    (93, 137),  (137, 227), (227, 93),  (15, 86),   (86, 85),   (85, 15),
    (129, 102), (102, 49),  (49, 129),  (14, 87),   (87, 86),   (86, 14),
    (55, 8),    (8, 9),     (9, 55),    (100, 47),  (47, 121),  (121, 100),
    (145, 23),  (23, 22),   (22, 145),  (88, 89),   (89, 179),  (179, 88),
    (6, 122),   (122, 196), (196, 6),   (88, 95),   (95, 96),   (96, 88),
    (138, 172), (172, 136), (136, 138), (215, 58),  (58, 172),  (172, 215),
    (115, 48),  (48, 219),  (219, 115), (42, 80),   (80, 81),   (81, 42),
    (195, 3),   (3, 51),    (51, 195),  (43, 146),  (146, 61),  (61, 43),
    (171, 175), (175, 199), (199, 171), (81, 82),   (82, 38),   (38, 81),
    (53, 46),   (46, 225),  (225, 53),  (144, 163), (163, 110), (110, 144),
    (52, 65),   (65, 66),   (66, 52),   (229, 228), (228, 117), (117, 229),
    (34, 127),  (127, 234), (234, 34),  (107, 108), (108, 69),  (69, 107),
    (109, 108), (108, 151), (151, 109), (48, 64),   (64, 235),  (235, 48),
    (62, 78),   (78, 191),  (191, 62),  (129, 209), (209, 126), (126, 129),
    (111, 35),  (35, 143),  (143, 111), (117, 123), (123, 50),  (50, 117),
    (222, 65),  (65, 52),   (52, 222),  (19, 125),  (125, 141), (141, 19),
    (221, 55),  (55, 65),   (65, 221),  (3, 195),   (195, 197), (197, 3),
    (25, 7),    (7, 33),    (33, 25),   (220, 237), (237, 44),  (44, 220),
    (70, 71),   (71, 139),  (139, 70),  (122, 193), (193, 245), (245, 122),
    (247, 130), (130, 33),  (33, 247),  (71, 21),   (21, 162),  (162, 71),
    (170, 169), (169, 150), (150, 170), (188, 174), (174, 196), (196, 188),
    (216, 186), (186, 92),  (92, 216),  (2, 97),    (97, 167),  (167, 2),
    (141, 125), (125, 241), (241, 141), (164, 167), (167, 37),  (37, 164),
    (72, 38),   (38, 12),   (12, 72),   (38, 82),   (82, 13),   (13, 38),
    (63, 68),   (68, 71),   (71, 63),   (226, 35),  (35, 111),  (111, 226),
    (101, 50),  (50, 205),  (205, 101), (206, 92),  (92, 165),  (165, 206),
    (209, 198), (198, 217), (217, 209), (165, 167), (167, 97),  (97, 165),
    (220, 115), (115, 218), (218, 220), (133, 112), (112, 243), (243, 133),
    (239, 238), (238, 241), (241, 239), (214, 135), (135, 169), (169, 214),
    (190, 173), (173, 133), (133, 190), (171, 208), (208, 32),  (32, 171),
    (125, 44),  (44, 237),  (237, 125), (86, 87),   (87, 178),  (178, 86),
    (85, 86),   (86, 179),  (179, 85),  (84, 85),   (85, 180),  (180, 84),
    (83, 84),   (84, 181),  (181, 83),  (201, 83),  (83, 182),  (182, 201),
    (137, 93),  (93, 132),  (132, 137), (76, 62),   (62, 183),  (183, 76),
    (61, 76),   (76, 184),  (184, 61),  (57, 61),   (61, 185),  (185, 57),
    (212, 57),  (57, 186),  (186, 212), (214, 207), (207, 187), (187, 214),
    (34, 143),  (143, 156), (156, 34),  (79, 239),  (239, 237), (237, 79),
    (123, 137), (137, 177), (177, 123), (44, 1),    (1, 4),     (4, 44),
    (201, 194), (194, 32),  (32, 201),  (64, 102),  (102, 129), (129, 64),
    (213, 215), (215, 138), (138, 213), (59, 166),  (166, 219), (219, 59),
    (242, 99),  (99, 97),   (97, 242),  (2, 94),    (94, 141),  (141, 2),
    (75, 59),   (59, 235),  (235, 75),  (24, 110),  (110, 228), (228, 24),
    (25, 130),  (130, 226), (226, 25),  (23, 24),   (24, 229),  (229, 23),
    (22, 23),   (23, 230),  (230, 22),  (26, 22),   (22, 231),  (231, 26),
    (112, 26),  (26, 232),  (232, 112), (189, 190), (190, 243), (243, 189),
    (221, 56),  (56, 190),  (190, 221), (28, 56),   (56, 221),  (221, 28),
    (27, 28),   (28, 222),  (222, 27),  (29, 27),   (27, 223),  (223, 29),
    (30, 29),   (29, 224),  (224, 30),  (247, 30),  (30, 225),  (225, 247),
    (238, 79),  (79, 20),   (20, 238),  (166, 59),  (59, 75),   (75, 166),
    (60, 75),   (75, 240),  (240, 60),  (147, 177), (177, 215), (215, 147),
    (20, 79),   (79, 166),  (166, 20),  (187, 147), (147, 213), (213, 187),
    (112, 233), (233, 244), (244, 112), (233, 128), (128, 245), (245, 233),
    (128, 114), (114, 188), (188, 128), (114, 217), (217, 174), (174, 114),
    (131, 115), (115, 220), (220, 131), (217, 198), (198, 236), (236, 217),
    (198, 131), (131, 134), (134, 198), (177, 132), (132, 58),  (58, 177),
    (143, 35),  (35, 124),  (124, 143), (110, 163), (163, 7),   (7, 110),
    (228, 110), (110, 25),  (25, 228),  (356, 389), (389, 368), (368, 356),
    (11, 302),  (302, 267), (267, 11),  (452, 350), (350, 349), (349, 452),
    (302, 303), (303, 269), (269, 302), (357, 343), (343, 277), (277, 357),
    (452, 453), (453, 357), (357, 452), (333, 332), (332, 297), (297, 333),
    (175, 152), (152, 377), (377, 175), (347, 348), (348, 330), (330, 347),
    (303, 304), (304, 270), (270, 303), (9, 336),   (336, 337), (337, 9),
    (278, 279), (279, 360), (360, 278), (418, 262), (262, 431), (431, 418),
    (304, 408), (408, 409), (409, 304), (310, 415), (415, 407), (407, 310),
    (270, 409), (409, 410), (410, 270), (450, 348), (348, 347), (347, 450),
    (422, 430), (430, 434), (434, 422), (313, 314), (314, 17),  (17, 313),
    (306, 307), (307, 375), (375, 306), (387, 388), (388, 260), (260, 387),
    (286, 414), (414, 398), (398, 286), (335, 406), (406, 418), (418, 335),
    (364, 367), (367, 416), (416, 364), (423, 358), (358, 327), (327, 423),
    (251, 284), (284, 298), (298, 251), (281, 5),   (5, 4),     (4, 281),
    (373, 374), (374, 253), (253, 373), (307, 320), (320, 321), (321, 307),
    (425, 427), (427, 411), (411, 425), (421, 313), (313, 18),  (18, 421),
    (321, 405), (405, 406), (406, 321), (320, 404), (404, 405), (405, 320),
    (315, 16),  (16, 17),   (17, 315),  (426, 425), (425, 266), (266, 426),
    (377, 400), (400, 369), (369, 377), (322, 391), (391, 269), (269, 322),
    (417, 465), (465, 464), (464, 417), (386, 257), (257, 258), (258, 386),
    (466, 260), (260, 388), (388, 466), (456, 399), (399, 419), (419, 456),
    (284, 332), (332, 333), (333, 284), (417, 285), (285, 8),   (8, 417),
    (346, 340), (340, 261), (261, 346), (413, 441), (441, 285), (285, 413),
    (327, 460), (460, 328), (328, 327), (355, 371), (371, 329), (329, 355),
    (392, 439), (439, 438), (438, 392), (382, 341), (341, 256), (256, 382),
    (429, 420), (420, 360), (360, 429), (364, 394), (394, 379), (379, 364),
    (277, 343), (343, 437), (437, 277), (443, 444), (444, 283), (283, 443),
    (275, 440), (440, 363), (363, 275), (431, 262), (262, 369), (369, 431),
    (297, 338), (338, 337), (337, 297), (273, 375), (375, 321), (321, 273),
    (450, 451), (451, 349), (349, 450), (446, 342), (342, 467), (467, 446),
    (293, 334), (334, 282), (282, 293), (458, 461), (461, 462), (462, 458),
    (276, 353), (353, 383), (383, 276), (308, 324), (324, 325), (325, 308),
    (276, 300), (300, 293), (293, 276), (372, 345), (345, 447), (447, 372),
    (352, 345), (345, 340), (340, 352), (274, 1),   (1, 19),    (19, 274),
    (456, 248), (248, 281), (281, 456), (436, 427), (427, 425), (425, 436),
    (381, 256), (256, 252), (252, 381), (269, 391), (391, 393), (393, 269),
    (200, 199), (199, 428), (428, 200), (266, 330), (330, 329), (329, 266),
    (287, 273), (273, 422), (422, 287), (250, 462), (462, 328), (328, 250),
    (258, 286), (286, 384), (384, 258), (265, 353), (353, 342), (342, 265),
    (387, 259), (259, 257), (257, 387), (424, 431), (431, 430), (430, 424),
    (342, 353), (353, 276), (276, 342), (273, 335), (335, 424), (424, 273),
    (292, 325), (325, 307), (307, 292), (366, 447), (447, 345), (345, 366),
    (271, 303), (303, 302), (302, 271), (423, 266), (266, 371), (371, 423),
    (294, 455), (455, 460), (460, 294), (279, 278), (278, 294), (294, 279),
    (271, 272), (272, 304), (304, 271), (432, 434), (434, 427), (427, 432),
    (272, 407), (407, 408), (408, 272), (394, 430), (430, 431), (431, 394),
    (395, 369), (369, 400), (400, 395), (334, 333), (333, 299), (299, 334),
    (351, 417), (417, 168), (168, 351), (352, 280), (280, 411), (411, 352),
    (325, 319), (319, 320), (320, 325), (295, 296), (296, 336), (336, 295),
    (319, 403), (403, 404), (404, 319), (330, 348), (348, 349), (349, 330),
    (293, 298), (298, 333), (333, 293), (323, 454), (454, 447), (447, 323),
    (15, 16),   (16, 315),  (315, 15),  (358, 429), (429, 279), (279, 358),
    (14, 15),   (15, 316),  (316, 14),  (285, 336), (336, 9),   (9, 285),
    (329, 349), (349, 350), (350, 329), (374, 380), (380, 252), (252, 374),
    (318, 402), (402, 403), (403, 318), (6, 197),   (197, 419), (419, 6),
    (318, 319), (319, 325), (325, 318), (367, 364), (364, 365), (365, 367),
    (435, 367), (367, 397), (397, 435), (344, 438), (438, 439), (439, 344),
    (272, 271), (271, 311), (311, 272), (195, 5),   (5, 281),   (281, 195),
    (273, 287), (287, 291), (291, 273), (396, 428), (428, 199), (199, 396),
    (311, 271), (271, 268), (268, 311), (283, 444), (444, 445), (445, 283),
    (373, 254), (254, 339), (339, 373), (282, 334), (334, 296), (296, 282),
    (449, 347), (347, 346), (346, 449), (264, 447), (447, 454), (454, 264),
    (336, 296), (296, 299), (299, 336), (338, 10),  (10, 151),  (151, 338),
    (278, 439), (439, 455), (455, 278), (292, 407), (407, 415), (415, 292),
    (358, 371), (371, 355), (355, 358), (340, 345), (345, 372), (372, 340),
    (346, 347), (347, 280), (280, 346), (442, 443), (443, 282), (282, 442),
    (19, 94),   (94, 370),  (370, 19),  (441, 442), (442, 295), (295, 441),
    (248, 419), (419, 197), (197, 248), (263, 255), (255, 359), (359, 263),
    (440, 275), (275, 274), (274, 440), (300, 383), (383, 368), (368, 300),
    (351, 412), (412, 465), (465, 351), (263, 467), (467, 466), (466, 263),
    (301, 368), (368, 389), (389, 301), (395, 378), (378, 379), (379, 395),
    (412, 351), (351, 419), (419, 412), (436, 426), (426, 322), (322, 436),
    (2, 164),   (164, 393), (393, 2),   (370, 462), (462, 461), (461, 370),
    (164, 0),   (0, 267),   (267, 164), (302, 11),  (11, 12),   (12, 302),
    (268, 12),  (12, 13),   (13, 268),  (293, 300), (300, 301), (301, 293),
    (446, 261), (261, 340), (340, 446), (330, 266), (266, 425), (425, 330),
    (426, 423), (423, 391), (391, 426), (429, 355), (355, 437), (437, 429),
    (391, 327), (327, 326), (326, 391), (440, 457), (457, 438), (438, 440),
    (341, 382), (382, 362), (362, 341), (459, 457), (457, 461), (461, 459),
    (434, 430), (430, 394), (394, 434), (414, 463), (463, 362), (362, 414),
    (396, 369), (369, 262), (262, 396), (354, 461), (461, 457), (457, 354),
    (316, 403), (403, 402), (402, 316), (315, 404), (404, 403), (403, 315),
    (314, 405), (405, 404), (404, 314), (313, 406), (406, 405), (405, 313),
    (421, 418), (418, 406), (406, 421), (366, 401), (401, 361), (361, 366),
    (306, 408), (408, 407), (407, 306), (291, 409), (409, 408), (408, 291),
    (287, 410), (410, 409), (409, 287), (432, 436), (436, 410), (410, 432),
    (434, 416), (416, 411), (411, 434), (264, 368), (368, 383), (383, 264),
    (309, 438), (438, 457), (457, 309), (352, 376), (376, 401), (401, 352),
    (274, 275), (275, 4),   (4, 274),   (421, 428), (428, 262), (262, 421),
    (294, 327), (327, 358), (358, 294), (433, 416), (416, 367), (367, 433),
    (289, 455), (455, 439), (439, 289), (462, 370), (370, 326), (326, 462),
    (2, 326),   (326, 370), (370, 2),   (305, 460), (460, 455), (455, 305),
    (254, 449), (449, 448), (448, 254), (255, 261), (261, 446), (446, 255),
    (253, 450), (450, 449), (449, 253), (252, 451), (451, 450), (450, 252),
    (256, 452), (452, 451), (451, 256), (341, 453), (453, 452), (452, 341),
    (413, 464), (464, 463), (463, 413), (441, 413), (413, 414), (414, 441),
    (258, 442), (442, 441), (441, 258), (257, 443), (443, 442), (442, 257),
    (259, 444), (444, 443), (443, 259), (260, 445), (445, 444), (444, 260),
    (467, 342), (342, 445), (445, 467), (459, 458), (458, 250), (250, 459),
    (289, 392), (392, 290), (290, 289), (290, 328), (328, 460), (460, 290),
    (376, 433), (433, 435), (435, 376), (250, 290), (290, 392), (392, 250),
    (411, 416), (416, 433), (433, 411), (341, 463), (463, 464), (464, 341),
    (453, 464), (464, 465), (465, 453), (357, 465), (465, 412), (412, 357),
    (343, 412), (412, 399), (399, 343), (360, 363), (363, 440), (440, 360),
    (437, 399), (399, 456), (456, 437), (420, 456), (456, 363), (363, 420),
    (401, 435), (435, 288), (288, 401), (372, 383), (383, 353), (353, 372),
    (339, 255), (255, 249), (249, 339), (448, 261), (261, 255), (255, 448),
    (133, 243), (243, 190), (190, 133), (133, 155), (155, 112), (112, 133),
    (33, 246),  (246, 247), (247, 33),  (33, 130),  (130, 25),  (25, 33),
    (398, 384), (384, 286), (286, 398), (362, 398), (398, 414), (414, 362),
    (362, 463), (463, 341), (341, 362), (263, 359), (359, 467), (467, 263),
    (263, 249), (249, 255), (255, 263), (466, 467), (467, 260), (260, 466),
    (75, 60),   (60, 166),  (166, 75),  (238, 239), (239, 79),  (79, 238),
    (162, 127), (127, 139), (139, 162), (72, 11),   (11, 37),   (37, 72),
    (121, 232), (232, 120), (120, 121), (73, 72),   (72, 39),   (39, 73),
    (114, 128), (128, 47),  (47, 114),  (233, 232), (232, 128), (128, 233),
    (103, 104), (104, 67),  (67, 103),  (152, 175), (175, 148), (148, 152),
    (119, 118), (118, 101), (101, 119), (74, 73),   (73, 40),   (40, 74),
    (107, 9),   (9, 108),   (108, 107), (49, 48),   (48, 131),  (131, 49),
    (32, 194),  (194, 211), (211, 32),  (184, 74),  (74, 185),  (185, 184),
    (191, 80),  (80, 183),  (183, 191), (185, 40),  (40, 186),  (186, 185),
    (119, 230), (230, 118), (118, 119), (210, 202), (202, 214), (214, 210),
    (84, 83),   (83, 17),   (17, 84),   (77, 76),   (76, 146),  (146, 77),
    (161, 160), (160, 30),  (30, 161),  (190, 56),  (56, 173),  (173, 190),
    (182, 106), (106, 194), (194, 182), (138, 135), (135, 192), (192, 138),
    (129, 203), (203, 98),  (98, 129),  (54, 21),   (21, 68),   (68, 54),
    (5, 51),    (51, 4),    (4, 5),     (145, 144), (144, 23),  (23, 145),
    (90, 77),   (77, 91),   (91, 90),   (207, 205), (205, 187), (187, 207),
    (83, 201),  (201, 18),  (18, 83),   (181, 91),  (91, 182),  (182, 181),
    (180, 90),  (90, 181),  (181, 180), (16, 85),   (85, 17),   (17, 16),
    (205, 206), (206, 36),  (36, 205),  (176, 148), (148, 140), (140, 176),
    (165, 92),  (92, 39),   (39, 165),  (245, 193), (193, 244), (244, 245),
    (27, 159),  (159, 28),  (28, 27),   (30, 247),  (247, 161), (161, 30),
    (174, 236), (236, 196), (196, 174), (103, 54),  (54, 104),  (104, 103),
    (55, 193),  (193, 8),   (8, 55),    (111, 117), (117, 31),  (31, 111),
    (221, 189), (189, 55),  (55, 221),  (240, 98),  (98, 99),   (99, 240),
    (142, 126), (126, 100), (100, 142), (219, 166), (166, 218), (218, 219),
    (112, 155), (155, 26),  (26, 112),  (198, 209), (209, 131), (131, 198),
    (169, 135), (135, 150), (150, 169), (114, 47),  (47, 217),  (217, 114),
    (224, 223), (223, 53),  (53, 224),  (220, 45),  (45, 134),  (134, 220),
    (32, 211),  (211, 140), (140, 32),  (109, 67),  (67, 108),  (108, 109),
    (146, 43),  (43, 91),   (91, 146),  (231, 230), (230, 120), (120, 231),
    (113, 226), (226, 247), (247, 113), (105, 63),  (63, 52),   (52, 105),
    (241, 238), (238, 242), (242, 241), (124, 46),  (46, 156),  (156, 124),
    (95, 78),   (78, 96),   (96, 95),   (70, 46),   (46, 63),   (63, 70),
    (116, 143), (143, 227), (227, 116), (116, 123), (123, 111), (111, 116),
    (1, 44),    (44, 19),   (19, 1),    (3, 236),   (236, 51),  (51, 3),
    (207, 216), (216, 205), (205, 207), (26, 154),  (154, 22),  (22, 26),
    (165, 39),  (39, 167),  (167, 165), (199, 200), (200, 208), (208, 199),
    (101, 36),  (36, 100),  (100, 101), (43, 57),   (57, 202),  (202, 43),
    (242, 20),  (20, 99),   (99, 242),  (56, 28),   (28, 157),  (157, 56),
    (124, 35),  (35, 113),  (113, 124), (29, 160),  (160, 27),  (27, 29),
    (211, 204), (204, 210), (210, 211), (124, 113), (113, 46),  (46, 124),
    (106, 43),  (43, 204),  (204, 106), (96, 62),   (62, 77),   (77, 96),
    (227, 137), (137, 116), (116, 227), (73, 41),   (41, 72),   (72, 73),
    (36, 203),  (203, 142), (142, 36),  (235, 64),  (64, 240),  (240, 235),
    (48, 49),   (49, 64),   (64, 48),   (42, 41),   (41, 74),   (74, 42),
    (214, 212), (212, 207), (207, 214), (183, 42),  (42, 184),  (184, 183),
    (210, 169), (169, 211), (211, 210), (140, 170), (170, 176), (176, 140),
    (104, 105), (105, 69),  (69, 104),  (193, 122), (122, 168), (168, 193),
    (50, 123),  (123, 187), (187, 50),  (89, 96),   (96, 90),   (90, 89),
    (66, 65),   (65, 107),  (107, 66),  (179, 89),  (89, 180),  (180, 179),
    (119, 101), (101, 120), (120, 119), (68, 63),   (63, 104),  (104, 68),
    (234, 93),  (93, 227),  (227, 234), (16, 15),   (15, 85),   (85, 16),
    (209, 129), (129, 49),  (49, 209),  (15, 14),   (14, 86),   (86, 15),
    (107, 55),  (55, 9),    (9, 107),   (120, 100), (100, 121), (121, 120),
    (153, 145), (145, 22),  (22, 153),  (178, 88),  (88, 179),  (179, 178),
    (197, 6),   (6, 196),   (196, 197), (89, 88),   (88, 96),   (96, 89),
    (135, 138), (138, 136), (136, 135), (138, 215), (215, 172), (172, 138),
    (218, 115), (115, 219), (219, 218), (41, 42),   (42, 81),   (81, 41),
    (5, 195),   (195, 51),  (51, 5),    (57, 43),   (43, 61),   (61, 57),
    (208, 171), (171, 199), (199, 208), (41, 81),   (81, 38),   (38, 41),
    (224, 53),  (53, 225),  (225, 224), (24, 144),  (144, 110), (110, 24),
    (105, 52),  (52, 66),   (66, 105),  (118, 229), (229, 117), (117, 118),
    (227, 34),  (34, 234),  (234, 227), (66, 107),  (107, 69),  (69, 66),
    (10, 109),  (109, 151), (151, 10),  (219, 48),  (48, 235),  (235, 219),
    (183, 62),  (62, 191),  (191, 183), (142, 129), (129, 126), (126, 142),
    (116, 111), (111, 143), (143, 116), (118, 117), (117, 50),  (50, 118),
    (223, 222), (222, 52),  (52, 223),  (94, 19),   (19, 141),  (141, 94),
    (222, 221), (221, 65),  (65, 222),  (196, 3),   (3, 197),   (197, 196),
    (45, 220),  (220, 44),  (44, 45),   (156, 70),  (70, 139),  (139, 156),
    (188, 122), (122, 245), (245, 188), (139, 71),  (71, 162),  (162, 139),
    (149, 170), (170, 150), (150, 149), (122, 188), (188, 196), (196, 122),
    (206, 216), (216, 92),  (92, 206),  (164, 2),   (2, 167),   (167, 164),
    (242, 141), (141, 241), (241, 242), (0, 164),   (164, 37),  (37, 0),
    (11, 72),   (72, 12),   (12, 11),   (12, 38),   (38, 13),   (13, 12),
    (70, 63),   (63, 71),   (71, 70),   (31, 226),  (226, 111), (111, 31),
    (36, 101),  (101, 205), (205, 36),  (203, 206), (206, 165), (165, 203),
    (126, 209), (209, 217), (217, 126), (98, 165),  (165, 97),  (97, 98),
    (237, 220), (220, 218), (218, 237), (237, 239), (239, 241), (241, 237),
    (210, 214), (214, 169), (169, 210), (140, 171), (171, 32),  (32, 140),
    (241, 125), (125, 237), (237, 241), (179, 86),  (86, 178),  (178, 179),
    (180, 85),  (85, 179),  (179, 180), (181, 84),  (84, 180),  (180, 181),
    (182, 83),  (83, 181),  (181, 182), (194, 201), (201, 182), (182, 194),
    (177, 137), (137, 132), (132, 177), (184, 76),  (76, 183),  (183, 184),
    (185, 61),  (61, 184),  (184, 185), (186, 57),  (57, 185),  (185, 186),
    (216, 212), (212, 186), (186, 216), (192, 214), (214, 187), (187, 192),
    (139, 34),  (34, 156),  (156, 139), (218, 79),  (79, 237),  (237, 218),
    (147, 123), (123, 177), (177, 147), (45, 44),   (44, 4),    (4, 45),
    (208, 201), (201, 32),  (32, 208),  (98, 64),   (64, 129),  (129, 98),
    (192, 213), (213, 138), (138, 192), (235, 59),  (59, 219),  (219, 235),
    (141, 242), (242, 97),  (97, 141),  (97, 2),    (2, 141),   (141, 97),
    (240, 75),  (75, 235),  (235, 240), (229, 24),  (24, 228),  (228, 229),
    (31, 25),   (25, 226),  (226, 31),  (230, 23),  (23, 229),  (229, 230),
    (231, 22),  (22, 230),  (230, 231), (232, 26),  (26, 231),  (231, 232),
    (233, 112), (112, 232), (232, 233), (244, 189), (189, 243), (243, 244),
    (189, 221), (221, 190), (190, 189), (222, 28),  (28, 221),  (221, 222),
    (223, 27),  (27, 222),  (222, 223), (224, 29),  (29, 223),  (223, 224),
    (225, 30),  (30, 224),  (224, 225), (113, 247), (247, 225), (225, 113),
    (99, 60),   (60, 240),  (240, 99),  (213, 147), (147, 215), (215, 213),
    (60, 20),   (20, 166),  (166, 60),  (192, 187), (187, 213), (213, 192),
    (243, 112), (112, 244), (244, 243), (244, 233), (233, 245), (245, 244),
    (245, 128), (128, 188), (188, 245), (188, 114), (114, 174), (174, 188),
    (134, 131), (131, 220), (220, 134), (174, 217), (217, 236), (236, 174),
    (236, 198), (198, 134), (134, 236), (215, 177), (177, 58),  (58, 215),
    (156, 143), (143, 124), (124, 156), (25, 110),  (110, 7),   (7, 25),
    (31, 228),  (228, 25),  (25, 31),   (264, 356), (356, 368), (368, 264),
    (0, 11),    (11, 267),  (267, 0),   (451, 452), (452, 349), (349, 451),
    (267, 302), (302, 269), (269, 267), (350, 357), (357, 277), (277, 350),
    (350, 452), (452, 357), (357, 350), (299, 333), (333, 297), (297, 299),
    (396, 175), (175, 377), (377, 396), (280, 347), (347, 330), (330, 280),
    (269, 303), (303, 270), (270, 269), (151, 9),   (9, 337),   (337, 151),
    (344, 278), (278, 360), (360, 344), (424, 418), (418, 431), (431, 424),
    (270, 304), (304, 409), (409, 270), (272, 310), (310, 407), (407, 272),
    (322, 270), (270, 410), (410, 322), (449, 450), (450, 347), (347, 449),
    (432, 422), (422, 434), (434, 432), (18, 313),  (313, 17),  (17, 18),
    (291, 306), (306, 375), (375, 291), (259, 387), (387, 260), (260, 259),
    (424, 335), (335, 418), (418, 424), (434, 364), (364, 416), (416, 434),
    (391, 423), (423, 327), (327, 391), (301, 251), (251, 298), (298, 301),
    (275, 281), (281, 4),   (4, 275),   (254, 373), (373, 253), (253, 254),
    (375, 307), (307, 321), (321, 375), (280, 425), (425, 411), (411, 280),
    (200, 421), (421, 18),  (18, 200),  (335, 321), (321, 406), (406, 335),
    (321, 320), (320, 405), (405, 321), (314, 315), (315, 17),  (17, 314),
    (423, 426), (426, 266), (266, 423), (396, 377), (377, 369), (369, 396),
    (270, 322), (322, 269), (269, 270), (413, 417), (417, 464), (464, 413),
    (385, 386), (386, 258), (258, 385), (248, 456), (456, 419), (419, 248),
    (298, 284), (284, 333), (333, 298), (168, 417), (417, 8),   (8, 168),
    (448, 346), (346, 261), (261, 448), (417, 413), (413, 285), (285, 417),
    (326, 327), (327, 328), (328, 326), (277, 355), (355, 329), (329, 277),
    (309, 392), (392, 438), (438, 309), (381, 382), (382, 256), (256, 381),
    (279, 429), (429, 360), (360, 279), (365, 364), (364, 379), (379, 365),
    (355, 277), (277, 437), (437, 355), (282, 443), (443, 283), (283, 282),
    (281, 275), (275, 363), (363, 281), (395, 431), (431, 369), (369, 395),
    (299, 297), (297, 337), (337, 299), (335, 273), (273, 321), (321, 335),
    (348, 450), (450, 349), (349, 348), (359, 446), (446, 467), (467, 359),
    (283, 293), (293, 282), (282, 283), (250, 458), (458, 462), (462, 250),
    (300, 276), (276, 383), (383, 300), (292, 308), (308, 325), (325, 292),
    (283, 276), (276, 293), (293, 283), (264, 372), (372, 447), (447, 264),
    (346, 352), (352, 340), (340, 346), (354, 274), (274, 19),  (19, 354),
    (363, 456), (456, 281), (281, 363), (426, 436), (436, 425), (425, 426),
    (380, 381), (381, 252), (252, 380), (267, 269), (269, 393), (393, 267),
    (421, 200), (200, 428), (428, 421), (371, 266), (266, 329), (329, 371),
    (432, 287), (287, 422), (422, 432), (290, 250), (250, 328), (328, 290),
    (385, 258), (258, 384), (384, 385), (446, 265), (265, 342), (342, 446),
    (386, 387), (387, 257), (257, 386), (422, 424), (424, 430), (430, 422),
    (445, 342), (342, 276), (276, 445), (422, 273), (273, 424), (424, 422),
    (306, 292), (292, 307), (307, 306), (352, 366), (366, 345), (345, 352),
    (268, 271), (271, 302), (302, 268), (358, 423), (423, 371), (371, 358),
    (327, 294), (294, 460), (460, 327), (331, 279), (279, 294), (294, 331),
    (303, 271), (271, 304), (304, 303), (436, 432), (432, 427), (427, 436),
    (304, 272), (272, 408), (408, 304), (395, 394), (394, 431), (431, 395),
    (378, 395), (395, 400), (400, 378), (296, 334), (334, 299), (299, 296),
    (6, 351),   (351, 168), (168, 6),   (376, 352), (352, 411), (411, 376),
    (307, 325), (325, 320), (320, 307), (285, 295), (295, 336), (336, 285),
    (320, 319), (319, 404), (404, 320), (329, 330), (330, 349), (349, 329),
    (334, 293), (293, 333), (333, 334), (366, 323), (323, 447), (447, 366),
    (316, 15),  (15, 315),  (315, 316), (331, 358), (358, 279), (279, 331),
    (317, 14),  (14, 316),  (316, 317), (8, 285),   (285, 9),   (9, 8),
    (277, 329), (329, 350), (350, 277), (253, 374), (374, 252), (252, 253),
    (319, 318), (318, 403), (403, 319), (351, 6),   (6, 419),   (419, 351),
    (324, 318), (318, 325), (325, 324), (397, 367), (367, 365), (365, 397),
    (288, 435), (435, 397), (397, 288), (278, 344), (344, 439), (439, 278),
    (310, 272), (272, 311), (311, 310), (248, 195), (195, 281), (281, 248),
    (375, 273), (273, 291), (291, 375), (175, 396), (396, 199), (199, 175),
    (312, 311), (311, 268), (268, 312), (276, 283), (283, 445), (445, 276),
    (390, 373), (373, 339), (339, 390), (295, 282), (282, 296), (296, 295),
    (448, 449), (449, 346), (346, 448), (356, 264), (264, 454), (454, 356),
    (337, 336), (336, 299), (299, 337), (337, 338), (338, 151), (151, 337),
    (294, 278), (278, 455), (455, 294), (308, 292), (292, 415), (415, 308),
    (429, 358), (358, 355), (355, 429), (265, 340), (340, 372), (372, 265),
    (352, 346), (346, 280), (280, 352), (295, 442), (442, 282), (282, 295),
    (354, 19),  (19, 370),  (370, 354), (285, 441), (441, 295), (295, 285),
    (195, 248), (248, 197), (197, 195), (457, 440), (440, 274), (274, 457),
    (301, 300), (300, 368), (368, 301), (417, 351), (351, 465), (465, 417),
    (251, 301), (301, 389), (389, 251), (394, 395), (395, 379), (379, 394),
    (399, 412), (412, 419), (419, 399), (410, 436), (436, 322), (322, 410),
    (326, 2),   (2, 393),   (393, 326), (354, 370), (370, 461), (461, 354),
    (393, 164), (164, 267), (267, 393), (268, 302), (302, 12),  (12, 268),
    (312, 268), (268, 13),  (13, 312),  (298, 293), (293, 301), (301, 298),
    (265, 446), (446, 340), (340, 265), (280, 330), (330, 425), (425, 280),
    (322, 426), (426, 391), (391, 322), (420, 429), (429, 437), (437, 420),
    (393, 391), (391, 326), (326, 393), (344, 440), (440, 438), (438, 344),
    (458, 459), (459, 461), (461, 458), (364, 434), (434, 394), (394, 364),
    (428, 396), (396, 262), (262, 428), (274, 354), (354, 457), (457, 274),
    (317, 316), (316, 402), (402, 317), (316, 315), (315, 403), (403, 316),
    (315, 314), (314, 404), (404, 315), (314, 313), (313, 405), (405, 314),
    (313, 421), (421, 406), (406, 313), (323, 366), (366, 361), (361, 323),
    (292, 306), (306, 407), (407, 292), (306, 291), (291, 408), (408, 306),
    (291, 287), (287, 409), (409, 291), (287, 432), (432, 410), (410, 287),
    (427, 434), (434, 411), (411, 427), (372, 264), (264, 383), (383, 372),
    (459, 309), (309, 457), (457, 459), (366, 352), (352, 401), (401, 366),
    (1, 274),   (274, 4),   (4, 1),     (418, 421), (421, 262), (262, 418),
    (331, 294), (294, 358), (358, 331), (435, 433), (433, 367), (367, 435),
    (392, 289), (289, 439), (439, 392), (328, 462), (462, 326), (326, 328),
    (94, 2),    (2, 370),   (370, 94),  (289, 305), (305, 455), (455, 289),
    (339, 254), (254, 448), (448, 339), (359, 255), (255, 446), (446, 359),
    (254, 253), (253, 449), (449, 254), (253, 252), (252, 450), (450, 253),
    (252, 256), (256, 451), (451, 252), (256, 341), (341, 452), (452, 256),
    (414, 413), (413, 463), (463, 414), (286, 441), (441, 414), (414, 286),
    (286, 258), (258, 441), (441, 286), (258, 257), (257, 442), (442, 258),
    (257, 259), (259, 443), (443, 257), (259, 260), (260, 444), (444, 259),
    (260, 467), (467, 445), (445, 260), (309, 459), (459, 250), (250, 309),
    (305, 289), (289, 290), (290, 305), (305, 290), (290, 460), (460, 305),
    (401, 376), (376, 435), (435, 401), (309, 250), (250, 392), (392, 309),
    (376, 411), (411, 433), (433, 376), (453, 341), (341, 464), (464, 453),
    (357, 453), (453, 465), (465, 357), (343, 357), (357, 412), (412, 343),
    (437, 343), (343, 399), (399, 437), (344, 360), (360, 440), (440, 344),
    (420, 437), (437, 456), (456, 420), (360, 420), (420, 363), (363, 360),
    (361, 401), (401, 288), (288, 361), (265, 372), (372, 353), (353, 265),
    (390, 339), (339, 249), (249, 390), (339, 448), (448, 255), (255, 339)]
