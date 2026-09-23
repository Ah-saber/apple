import unittest
from ir_sr.training import validate_resume_configuration, learning_rate
class ContinuationTests(unittest.TestCase):
 def setUp(self):
  self.old=dict(max_steps=8000,max_wall_seconds=3600,test_every=8000,version='pilot',lr_schedule_steps=200000,warmup_steps=200,lr=.0002,min_lr=.000002,batch_size=16,seed=928,normalization='none')
  self.new=dict(self.old,max_steps=200000,max_wall_seconds=21600,test_every=200000,version='full')
 def test_allow_frozen_schedule_extension(self):
  validate_resume_configuration(self.old,self.new,dict(step=8000),True)
  for step in (8001,10000,100000,200000):self.assertEqual(learning_rate(step,self.old),learning_rate(step,self.new))
 def test_explicit_opt_in_required(self):
  with self.assertRaises(ValueError):validate_resume_configuration(self.old,self.new,dict(step=8000))
 def test_reject_training_change_and_incomplete_parent(self):
  for key,value in [('seed',929),('batch_size',8),('normalization','layernorm'),('lr_schedule_steps',300000)]:
   with self.subTest(key=key):
    with self.assertRaises(ValueError):validate_resume_configuration(self.old,dict(self.new,**{key:value}),dict(step=8000),True)
  with self.assertRaises(ValueError):validate_resume_configuration(self.old,self.new,dict(step=7999),True)
  with self.assertRaises(ValueError):validate_resume_configuration(self.old,dict(self.new,max_steps=200001,test_every=200001),dict(step=8000),True)
if __name__=='__main__':unittest.main()
