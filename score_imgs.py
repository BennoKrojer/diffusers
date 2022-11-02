import torch
import glob
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--verbose',action='store_true')
parser.add_argument('--img_latent_dir',type=str)
parser.add_argument('--prompt_latent_path',type=str)
args = parser.parse_args()

if not args.img_latent_dir:
    args.img_latent_dir = '/home/krojerb/img-gen-project/diffusers/animals_latent'
if not args.prompt_latent_path:
    args.prompt_latent_path = '/home/krojerb/img-gen-project/diffusers/queries_latent/dog_latent.pt'


# get all latent image pickles
images = {}
for path in glob.glob(f"{args.img_latent_dir}/*"):
    # fname = 
    images[path] = torch.load(path, map_location=torch.device(device))

# get the query latent pickle
prompt = torch.load(args.prompt_latent_path, map_location=torch.device(device)).flatten()
query = args.prompt_latent_path.split('/')[-1].split('.')[0].split('_')[0]

# flatten and compare
similarities = {}
cos = torch.nn.CosineSimilarity(dim=0)
for path, vect in images.items():
    fvect = vect.flatten()
    sim =  cos(prompt,fvect)
    similarities[sim] = path
    
if args.verbose:
    for (sim,img_path) in similarities.items():
        print(img_path.split('/')[-1].split('.')[0],sim.item())
# return cosine sim
print(f" the image with the highest similarity to the query '{query}' is {similarities[max(similarities)]}")
