import pandas as pd
from difflib import SequenceMatcher

df_reference_identifier = pd.read_csv('C:/Users/Hussein/Desktop/meron_time_linkage/isiolo_meron_supervisor_qualitative_identifier.csv')
df_reference_members = pd.read_csv('C:/Users/Hussein/Desktop/meron_time_linkage/isiolo_meron_supervisor_qualitative_members.csv')
df_reference_anthros = pd.read_csv('C:/Users/Hussein/Desktop/meron_time_linkage/isiolo_meron_supervisor_qualitative_anthros.csv')
df_reference_members_combined = df_reference_members.merge(df_reference_anthros, how='inner', on=['_parent_index', '_index'])
df_reference_members_complete = df_reference_members_combined.merge(df_reference_identifier, how='inner', on=['_parent_index'])

dict_linkup_variables = {
    'cluster_no' : 'Section_1_HH_Location/smart_identification_id/cluster_no',
    'team_no' : 'Section_1_HH_Location/smart_identification_id/team_no',
    'hh_no' : 'Section_1_HH_Location/smart_identification_id/hh_no',
    #'age' : 'section_2_child_details/repeat_1/age',
    #'member_name' : 'section_2_child_details/repeat_1/child_name'
}

for key, value in dict_linkup_variables.iteritems():
    df_reference_members_complete.rename(columns={value: key}, inplace=True)

df_original = pd.read_csv('C:/Users/Hussein/Desktop/smart_data/isiolo_smart.csv')

#df_backcheck_merge = df_reference_members_complete.merge(df_original, how='inner', on=['cluster_no', 'team_no', 'hh_no'])

df_backcheck_merge = df_original.merge(df_reference_members_complete, how='inner', on=['cluster_no', 'team_no', 'hh_no'])



#limit columns

df_backcheck_merge = df_backcheck_merge[['cluster_no', 'hh_no', 'team_no', 'member_name', 'section_2_child_details/repeat_1/child_name',
                                         'section_3_smart_anthro/repeat_2/height', 'height',
                                         'section_3_smart_anthro/repeat_2/weight', 'weight',
                                         'section_3_smart_anthro/repeat_2/muac', 'muac']]

df_backcheck_merge.rename(columns={'member_name': 'member_name_SMART',
                                   'section_2_child_details/repeat_1/child_name': 'member_name_BACKCHECK',
                                   'section_3_smart_anthro/repeat_2/height' : 'height_BACKCHECK',
                                   'height' : 'height_SMART',
                                   'section_3_smart_anthro/repeat_2/weight' : 'weight_BACKCHECK',
                                   'weight' : 'weight_SMART',
                                   'muac' : 'muac_SMART',
                                   'section_3_smart_anthro/repeat_2/muac' : 'muac_BACKCHECK'}, inplace=True)

df_backcheck_merge.muac_BACKCHECK = df_backcheck_merge.muac_BACKCHECK * 10

df_backcheck_merge['Difference'] = 0.5
df_backcheck_merge['weight_is_correct'] = False
df_backcheck_merge['height_is_correct'] = False
df_backcheck_merge['muac_is_correct'] = False
df_backcheck_merge['all_correct'] = False

for index, row in df_backcheck_merge.iterrows():
    val = SequenceMatcher(None, row['member_name_BACKCHECK'], row['member_name_SMART']).ratio()
    df_backcheck_merge.set_value(index, 'Difference', 1 - val)
    val_height_smart = row['height_SMART']
    val_height_backcheck = row['height_BACKCHECK']
    val_weight_smart = row['weight_SMART']
    val_weight_backcheck = row['weight_BACKCHECK']
    val_muac_smart = row['muac_SMART']
    val_muac_backcheck = row['muac_BACKCHECK']
    if val_height_smart == val_height_backcheck:
        df_backcheck_merge.set_value(index, 'height_is_correct', True)
    if val_weight_smart == val_weight_backcheck:
        df_backcheck_merge.set_value(index, 'weight_is_correct', True)
    if val_muac_smart == val_muac_backcheck:
        df_backcheck_merge.set_value(index, 'muac_is_correct', True)
    if (val_height_backcheck == val_height_smart) and (val_weight_backcheck == val_weight_smart) and (val_muac_backcheck == val_muac_smart):
        df_backcheck_merge.set_value(index, 'all_correct', True)
    #row['Difference'] = SequenceMatcher(None, 'member_name_BACKCHECK', 'member_name_SMART').ratio()

#query = 'Difference >= 0.5'
#df_backcheck_merge = df_backcheck_merge.query(query)

df_backcheck_merge = df_backcheck_merge.sort_values(by=['cluster_no', 'team_no', 'hh_no', 'Difference'])


writer = pd.ExcelWriter('C:/Users/Hussein/Desktop/output.xls')
df_backcheck_merge.to_excel(writer, 'Sheet1')
df_reference_members_complete.to_excel(writer, 'Sheet2')
df_reference_members_combined.to_excel(writer, 'Sheet3')
df_reference_anthros.to_excel(writer, 'Sheet4')
df_reference_members.to_excel(writer, 'Sheet5')
writer.save()


