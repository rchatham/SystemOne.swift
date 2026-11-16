import json
import unittest
from harness import HTTPModel, ScriptedModel, evaluate, loads_response, summarize, execute

CASE={'id':'time','family':'direct','messages':[{'role':'user','content':'Time?'}],'expected':['current_time','generate_text','finish']}
class Tests(unittest.TestCase):
 def test_success(self):
  self.assertTrue(evaluate(CASE,ScriptedModel(CASE['expected']),1)['passed'])
 def test_wrong_tool(self):
  r=evaluate(CASE,ScriptedModel(['list_notes','generate_text','finish']),1)
  self.assertIn('wrong_tool',r['errors']); self.assertFalse(r['passed'])
 def test_repeat(self):
  r=evaluate(CASE,ScriptedModel(['current_time']*10),1,4)
  self.assertIn('repeated_tool',r['errors']); self.assertEqual(r['status'],'max_steps')
 def test_early_stop(self):
  self.assertIn('premature_or_wrong_finish',evaluate(CASE,ScriptedModel(['finish']),1)['errors'])
 def test_errors_excluded(self):
  class Broken:
   def decide(self,*args): raise TimeoutError('timeout')
  r=evaluate(CASE,Broken(),1)
  s=summarize([r]); self.assertEqual(s['evaluated'],0);self.assertIsNone(s['pass_rate'])
 def test_no_oracle_in_model_input(self):
  class Spy:
   def decide(self,state,options):
    assert 'expected' not in json.dumps(state)
    return 'finish',None,{}
  evaluate(CASE,Spy(),1)
 def test_post_tool_state(self):
  class Spy:
   def decide(self,state,options):
    assert state['observations'][0]['action']=='current_time'
    return 'finish',None,{}
  evaluate({**CASE,'initial_observations':[{'action':'current_time','result':{}}]},Spy(),1)
 def test_invalid_probabilities(self):
  with self.assertRaises(ValueError): loads_response({'answers':{'action':{'choice':'x','probabilities':{'x':float('nan'),'y':0}}}},['x','y'])
 def test_shuffle_and_instability(self):
  class First:
   def decide(self,state,options): return options[0],None,{}
  rs=[evaluate(CASE,First(),s,1) for s in range(10)]
  self.assertIn('time',summarize(rs)['inconsistent_cases'])
 def test_mutation(self):
  world={'notes':[],'events':[]}
  execute('save_note',{'text':'hello'},world)
  self.assertEqual(execute('list_notes',{},world)['notes'],['hello'])
 def test_http_contract(self):
  from unittest.mock import patch
  import io
  def respond(req, timeout):
   data=json.loads(req.data)
   self.assertEqual(data['questions']['action']['type'],'choice')
   self.assertNotIn('expected',data['state'])
   return io.BytesIO(json.dumps({'answers':{'action':{'choice':'finish'}}}).encode())
  with patch('urllib.request.urlopen',side_effect=respond):
   r=evaluate(CASE,HTTPModel('http://localhost/v1/systemone'),1)
   self.assertEqual(r['actions'],['finish'])
   self.assertNotEqual(r['status'],'infrastructure_error')
if __name__=='__main__': unittest.main()
