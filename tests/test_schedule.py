import unittest
from PIL import Image
import torch
from src.runner import image_error
from src.lcm_math import scalings_for_boundary_conditions

class ScheduleTests(unittest.TestCase):
    def test_boundary_is_identity(self):
        skip,out=scalings_for_boundary_conditions(torch.tensor(0.).float())
        self.assertEqual(skip.item(),1.)
        self.assertEqual(out.item(),0.)
    def test_pixel_mse(self):
        self.assertEqual(image_error(Image.new("RGB",(2,2),"white"),Image.new("RGB",(2,2),"black")),1.)
