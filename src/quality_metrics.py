from fid_score.fid_score import FidScore
def calculate_FID(sample_path, test_sample_path, batch_size=100):
    fid = FidScore([test_sample_path, sample_path],device='cuda:0', batch_size=batch_size)
    score = fid.calculate_fid_score()
    return score

if __name__=='__main__':
    sample_path='FID_images/mnist__50_0.02_0.0_'
    test_sample_path='FID_images_real_test/mnist__50_0.02_0.0_'
    print(calculate_FID(sample_path, test_sample_path, batch_size=100))