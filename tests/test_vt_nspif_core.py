"""Deterministic software checks, not prescription performance results."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import vt_nspif_core as v

def entity(value, typ='drug_name', group='m1'):
    return {'type':typ,'value':value,'medication_group':group}

class ExactClient:
    def lookup(self, term):
        return {'status':'exact','rxcui':'123',
                'properties':{'payload':{'properties':{'name':term,'tty':'IN'}}}}

class CandidateClient:
    def lookup(self, term):
        return {'status':'unresolved','candidates':[{'rxcui':'123'}]}

class ProtocolTests(unittest.TestCase):
    def test_leading_decimal_not_lost(self):
        e=v.extract_group('Drug .5 mg', 'm1', {})
        self.assertEqual([x['value'] for x in e if x['type']=='strength'], ['.5 mg'])
    def test_decimal_formats_preserve_complete_span(self):
        for value in ['.5 mg', '0.5 mg', '2.50 mg', ',5 mg', '0,5 mg']:
            with self.subTest(value=value):
                text='Drug '+value
                es=v.extract_group(text,'m1',{})
                self.assertEqual([e['value'] for e in es],[value])
                a,b=es[0]['span'];self.assertEqual(text[a:b],value)
    def test_fraction_spacing_and_unicode_mixed(self):
        for value in ['1 / 2 tablet','1 1/2 tablets','1½ tablets','¼ tablet','¾ tablet']:
            with self.subTest(value=value):
                es=v.extract_group('take '+value,'m1',{})
                self.assertEqual([(e['type'],e['value']) for e in es],[('dose',value)])
    def test_signed_quantity_not_truncated(self):
        for value in ['-.5 mg','−.5 mg','+0.5 mg']:
            es=v.extract_group('Drug '+value,'m1',{})
            self.assertEqual(es[0]['value'],value)
            self.assertIn('signed_quantity',es[0]['numeric_review_flags'])
    def test_scientific_quantity_not_truncated(self):
        for value in ['5e-3 mg','2.5E+2 mg']:
            es=v.extract_group('Drug '+value,'m1',{})
            self.assertEqual(es[0]['value'],value)
            self.assertIn('scientific_quantity',es[0]['numeric_review_flags'])
    def test_range_not_reduced_to_last_number(self):
        for value in ['5-10 mg','.5–1 mg']:
            es=v.extract_group('take '+value,'m1',{})
            self.assertEqual(es[0]['value'],value)
            self.assertEqual(es[0]['type'],'dose')
            self.assertIn('quantity_range',es[0]['numeric_review_flags'])
    def test_decimal_comma_requires_review(self):
        es=v.extract_group('Drug 0,5 mg','m1',{})
        self.assertIn('decimal_comma_quantity',es[0]['numeric_review_flags'])
    def test_zero_denominator_requires_review(self):
        es=v.extract_group('take 1/0 tablet','m1',{})
        self.assertEqual(es[0]['value'],'1/0 tablet')
        self.assertIn('zero_denominator',es[0]['numeric_review_flags'])
    def test_malformed_decimal_not_suffix_matched(self):
        self.assertFalse(v.extract_group('Drug 1..5 mg','m1',{}))
    def test_unsupported_compound_unit_not_partial_strength(self):
        self.assertFalse(v.extract_group('Drug 5 mg/kg','m1',{}))
        self.assertFalse(v.extract_group('take 5 mg/kg','m1',{}))
    def test_numeric_flags_trigger_group_review(self):
        es=v.extract_group('Drug -.5 mg','m1',{'drug':{'rxcui':'123'}})
        result=v.validate_entities(es,ExactClient())
        self.assertTrue(result['decisions'][0]['review_required'])
        self.assertIn('signed_quantity',result['decisions'][0]['flags'])
    def test_decimal_concentration_retains_denominator(self):
        es=v.extract_group('Drug .5 mg/.5 mL take .5 mL','m1',{})
        self.assertEqual([(e['type'],e['value']) for e in es],
                         [('strength','.5 mg/.5 mL'),('dose','.5 mL')])
    def test_capsule_dose_separate_from_strength(self):
        es=v.extract_group('Drug 250 mg take 2 capsules','m1',{})
        self.assertEqual([(e['type'],e['value']) for e in es],
                         [('strength','250 mg'),('dose','2 capsules')])

    def test_liquid_is_administered_dose(self):
        e=v.extract_group('take 5 ml', 'm1', {})
        self.assertEqual([(x['type'],x['value']) for x in e], [('dose','5 ml')])
    def test_injection_is_administered_dose(self):
        e=v.extract_group('inject .5 mL SC', 'm1', {})
        self.assertIn(('dose','.5 mL'), [(x['type'],x['value']) for x in e])
    def test_fractions_and_source_spans(self):
        for value in ['1/2 tablet','1 1/2 tablets','½ tablet','0.5 tablet','0,5 tablet']:
            text='take '+value
            e=v.extract_group(text,'m1',{})
            self.assertEqual(e[0]['value'],value)
            a,b=e[0]['span'];self.assertEqual(text[a:b],value)
    def test_concentration_and_dose_are_separate(self):
        e=v.extract_group('Drug 250 mg/5 ml take 5 ml', 'm1', {})
        self.assertEqual([(x['type'],x['value']) for x in e],
                         [('strength','250 mg/5 ml'),('dose','5 ml')])
    def test_mass_dose_not_strength(self):
        e=v.extract_group('administer 5 mg IV','m1',{})
        self.assertEqual(e[0]['type'],'dose')
    def test_strength_preserved_without_dose(self):
        es=[entity('Drug'),entity('500 mg','strength')]
        r=v.make_r5_resource(es,'Patient/research','m1',False)
        self.assertIn('500 mg',r['medication']['concept']['text'])
        self.assertNotIn('dosageInstruction',r)
    def test_all_fields_preserved(self):
        es=[entity('Drug')]+[entity(t+' value',t) for t in v.ENTITY_TYPES if t!='drug_name']
        r=v.make_r5_resource(es,'Patient/research','m1',False)
        import json
        out=json.dumps(r)
        for e in es:self.assertIn(e['value'],out)
    def test_unmapped_field_flagged(self):
        r=v.make_r5_resource([entity('Drug'),entity('special','custom')],
                             'Patient/research','m1',False)
        self.assertTrue(any('No typed mapping' in n['text'] for n in r['note']))
    def test_unicode_and_decimal_preservation(self):
        self.assertEqual(v.normalize_text('ওষুধ  2.5 mg'), 'ওষুধ 2.5 mg')

    def test_substitution_is_fp_and_fn(self):
        c=v.entity_counts([entity('A')],[entity('B')])['per_type']['drug_name']
        self.assertEqual((c['TP'],c['FP'],c['FN']), (0,1,1))

    def test_duplicate_assertion_is_fp(self):
        c=v.entity_counts([entity('A')],[entity('A'),entity('A')])
        self.assertEqual(c['per_type']['drug_name']['FP'],1)

    def test_wrong_association_is_incorrect(self):
        c=v.entity_counts([entity('A')],[entity('A',group='m2')])
        self.assertFalse(c['record_exact'])
        self.assertEqual(c['per_type']['drug_name']['FN'],1)

    def test_undefined_precision_and_recall(self):
        self.assertIsNone(v.precision_recall_f1(0,0,0)['precision'])
        self.assertIsNone(v.precision_recall_f1(0,0,0)['recall'])

    def test_failed_nonempty_prediction_is_deletion(self):
        s=v.text_scores('abc','')
        self.assertEqual(s['CER'],1)
        self.assertEqual(s['character_counts']['D'],3)

    def test_insertion_error_can_exceed_one(self):
        self.assertGreater(v.text_scores('a','abcdef')['CER'],1)

    def test_empty_reference_not_perfect(self):
        self.assertIsNone(v.text_scores('','x')['CER'])

    def test_source_span_and_strength_dose(self):
        aliases={'amoxicillin':{'rxcui':'123'}}
        text='Amoxicillin 500 mg Take 2 tablets PO twice daily for 7 days'
        entities=v.extract_group(text,'m1',aliases)
        values={e['type']:e['value'] for e in entities}
        self.assertEqual(values['strength'],'500 mg')
        self.assertEqual(values['dose'],'2 tablets')
        for e in entities:
            a,b=e['span']
            self.assertEqual(text[a:b],e['value'])

    def test_unknown_drug_is_not_fuzzy_invented(self):
        e=v.extract_group('Unknownxyz 5 mg','m1',{'known':{'rxcui':'1'}})
        self.assertFalse(any(x['type']=='drug_name' for x in e))

    def test_approximate_candidate_requires_review(self):
        out=v.validate_entities([entity('A')],CandidateClient())
        self.assertTrue(out['decisions'][0]['review_required'])
        self.assertEqual(out['entities'][0]['value'],'A')
        self.assertFalse(out['value_changing_corrections'])

    def test_exact_lookup_retains_raw_value(self):
        out=v.validate_entities([entity('A')],ExactClient())
        self.assertFalse(out['decisions'][0]['review_required'])
        self.assertEqual(out['entities'][0]['value'],'A')

    def test_r5_requires_identity_and_does_not_impute(self):
        self.assertIsNone(v.make_r5_resource([entity('A')],None,'m1',False))
        resource=v.make_r5_resource([entity('A')],'Patient/research-1','m1',False)
        self.assertIn('medication',resource)
        self.assertNotIn('medicationCodeableConcept',resource)
        self.assertNotIn('dosageInstruction',resource)

    def test_review_withholds_resource(self):
        self.assertIsNone(v.make_r5_resource([entity('A')],'Patient/x','m1',True))

    def test_manifest_cannot_overwrite_split(self):
        with self.assertRaises(ValueError):
            v.assign_group_splits([{'split':'test','writer_id':'w1'}])

    def test_cache_offline_does_not_claim_nonexistent_drug(self):
        with tempfile.TemporaryDirectory() as temp:
            out=v.TerminologyClient(temp).get('version.json')
            self.assertEqual(out['status'],'unavailable_offline')

    def test_ablation_reuses_same_transcription(self):
        row={'record_id':'r1','source':'fixture','language':'en',
             'prescription_group':'p1','reference_text':'A 5 mg',
             'reference_entities':[entity('A'),entity('5 mg','strength')],
             'authentic':False,'split':'test','task':'provided_regions'}
        rec={'regions':[{'medication_group':'m1','text':'A 5 mg'}],
             'prediction':'A 5 mg','ocr_error':[], 'recognition_seconds':0}
        aliases={'a':{'rxcui':'123'}}
        base=v.score_configuration(row,rec,aliases,CandidateClient(),False)
        full=v.score_configuration(row,rec,aliases,CandidateClient(),True)
        self.assertEqual(base['prediction'],full['prediction'])
        self.assertEqual(base['entity_scores']['per_type']['drug_name']['TP'],1)
        self.assertEqual(full['entity_scores']['per_type']['drug_name']['FN'],1)
        self.assertEqual(full['accepted_tuples'],0)

    def test_no_model_validity_claim_when_absent(self):
        summary=v.conformance_summary([])
        self.assertIsNone(summary['package_model_valid_rate'])
        self.assertIsNone(summary['hl7_profile_conformant_resources'])

    def test_bootstrap_requires_actual_pairs(self):
        with self.assertRaises(ValueError):
            v.paired_bootstrap([{'record_id':'a','config':'left','group':'g'}],
                               'left','right',n=10)

if __name__ == '__main__':
    unittest.main(verbosity=2)
