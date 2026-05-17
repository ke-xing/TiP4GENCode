import os
import argparse
import imageio.v2 as imageio
import py360convert
import subprocess
import datetime
import shutil
from concurrent.futures import ThreadPoolExecutor,as_completed


def extract_frames(file_path, num_frames=8):
    # 初始化读取器并获取元数据
    reader = imageio.get_reader(file_path)
    
    try:
        # 尝试获取总帧数（适用于视频）
        total_frames = reader.count_frames()
    except (AttributeError, KeyError):
        # 遍历计算总帧数（适用于GIF）
        total_frames = 0
        for _ in reader:
            total_frames += 1
        reader.close()  # 关闭后重新打开
        reader = imageio.get_reader(file_path)
    
    # 计算帧间隔，确保至少取1帧[2,7](@ref)
    interval = max(1, total_frames // num_frames)
    
    frames = []
    for i, frame in enumerate(reader):
        # 均匀采样帧序列[1,5](@ref)
        if i % interval == 0:
            frames.append(frame)
            if len(frames) >= num_frames:
                break
    
    reader.close()
    return frames

parser=argparse.ArgumentParser()

parser.add_argument('--input_path')
parser.add_argument('--outname',default='')
parser.add_argument('--post',default='')

args=parser.parse_args()


post=args.post
data_path=None
per_data_path=None
img_dirs=[]


if os.path.isfile(args.input_path):
    path=args.input_path
    video_path = path 
    base_path=os.path.dirname(path)
    img_dir_name=os.path.splitext(os.path.basename(path))[0]
    img_path=os.path.join(base_path,img_dir_name)
    os.makedirs(img_path,exist_ok=True)
    frames = extract_frames(video_path,num_frames=8)

    for i, frame in enumerate(frames):
        imageio.imwrite(f"{img_path}/frame_{i:03d}.png", frame)
    
    os.remove(video_path)
    base=img_dir_name
    parent=os.path.basename(os.path.dirname(args.input_path))
    per_imgs_path=f'{parent}/{base}'

    
    os.makedirs(per_imgs_path,exist_ok=True)
    
    data_path=img_path
    per_data_path=per_imgs_path
    
    for file in os.listdir(data_path):
        if os.path.isfile(os.path.join(data_path,file)):
            file_path=os.path.join(data_path,file)
            per_path=f'{per_imgs_path}/{file}'
            
            img=imageio.imread(file_path)
            
            per_img = py360convert.e2p(
            e_img=img,
            fov_deg=(90, 90), 
            u_deg=0,  
            v_deg=0,  
            out_hw=(512,512),
            mode='bilinear'
            )
            
            imageio.imsave(per_path,per_img)
            
            file_dirs=os.path.join(data_path,os.path.splitext(file)[0])
            os.makedirs(file_dirs,exist_ok=True)
            shutil.copy(file_path,f'{file_dirs}/{file}')
            img_dirs.append(file_dirs)
   
else:
    
    base=os.path.basename(args.input_path)
    parent=os.path.basename(os.path.dirname(args.input_path))
    per_imgs_path=f'data/per/{parent}/{base}'
    

    
    os.makedirs(per_imgs_path,exist_ok=True)
    
    data_path=args.input_path
    per_data_path=per_imgs_path
    
    for file in os.listdir(args.input_path):
        if os.path.isfile(os.path.join(args.input_path,file)):
            
            file_path=os.path.join(args.input_path,file)

            img=imageio.imread(file_path)
        
            per_img = py360convert.e2p(
            e_img=img,
            fov_deg=(90, 90), 
            u_deg=0,  
            v_deg=0,  
            out_hw=(512,512),
            mode='bilinear'
            )
            per_path=f'{per_imgs_path}/{file}'
            imageio.imsave(per_path,per_img)
            
            file_dirs=os.path.join(args.input_path,os.path.splitext(file)[0])
            os.makedirs(file_dirs,exist_ok=True)
            
            shutil.copy(file_path,f'{file_dirs}/{file}')
            img_dirs.append(file_dirs)




data_path=os.path.abspath(data_path)
per_data_path=os.path.abspath(per_data_path)
data_name=os.path.basename(data_path)
parent_name=os.path.basename(os.path.dirname(data_path))

current_date = datetime.date.today()


formatted_date = current_date.strftime("%m_%d_%Y")

print(per_data_path)



if post=='train':
    print('estimating pose...')
    command=f'conda run -n monst3r python ./monst3r/demo.py --input {per_data_path} --output_dir {os.path.dirname(per_data_path)} --seq_name {os.path.basename(per_data_path)}'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,cwd='./monst3r')

    # 读取输出
    stdout, stderr = process.communicate()

    # 打印标准输出
    print("标准输出:")
    print(stdout)

    # 打印标准错误（如果有）
    if stderr:
        print("标准错误:")
        print(stderr)
    print('done')


def run_script0(img_dir,frame,stage):
    env = os.environ.copy()
    gpus=['0,''1','2','3','4','5','6','7']
    env['CUDA_VISIBLE_DEVICES'] = gpus[frame%8]
    
    command=f'python train.py -s {img_dir} -m output/4d_exp/{parent_name}/{data_name}/{os.path.basename(img_dir)} --frame {frame} --stage {stage}'
    subprocess.run(command, env=env,shell=True)


def run_script(img_dir,frame,post):
    env = os.environ.copy()
    gpus=['0','1','2','3','4','5','6','7']
    env['CUDA_VISIBLE_DEVICES'] = gpus[frame%4]
    
    if post=='train':
        command=f'python train.py -s {img_dir} -m output/4d_exp/{parent_name}/{data_name}/{os.path.basename(img_dir)} --frame {frame} --stage 2'
        subprocess.run(command, env=env,shell=True)
        
        command=f'python render_c.py -s {img_dir} -m output/4d_exp/{parent_name}/{data_name}/{os.path.basename(img_dir)} --iteration 15000 --frame {frame}'
        subprocess.run(command, env=env,shell=True)
    else:
        command=f'python render_c.py -s {img_dir} -m output/4d_exp/{parent_name}/{data_name}/{os.path.basename(img_dir)} --iteration 10000 --frame {frame}'
        subprocess.run(command, env=env,shell=True)

def assign_scripts_to_gpus(scripts,post):
    
    if post=='train':
        run_script0(scripts[0],0,0)

        
        with ThreadPoolExecutor(max_workers=8) as executor:

            futures = [executor.submit(run_script0, script,i,1) for i,script in enumerate(scripts)]
        
            for future in as_completed(futures):
                try:
                    future.result()
                    print(f"Script on GPU completed successfully.")
                except Exception as exc:
                    print(f"Script on GPU generated an exception: {exc}")
    
    with ThreadPoolExecutor(max_workers=4) as executor:

        futures = [executor.submit(run_script, script,i,post) for i,script in enumerate(scripts)]
    
        for future in as_completed(futures):
            try:
                future.result()
                print(f"Script on GPU completed successfully.")
            except Exception as exc:
                print(f"Script on GPU generated an exception: {exc}")



assign_scripts_to_gpus(sorted(img_dirs),post=post)


