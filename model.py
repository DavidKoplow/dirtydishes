import modal
from FastSAM.fastsam import FastSAM, FastSAMPrompt

stub = modal.Stub("example-get-started")


@stub.function()
def square(x):
    print("This code is running on a remote worker!")
    return x**2

@stub.function()
def compute_segment(prompt_process : FastSAMPrompt):
    ann = prompt_process.everything_prompt()
    return ann


@stub.local_entrypoint()
def main():
    DEVICE = "cpu"
    model = FastSAM("./FastSAM/FastSAM.pt")
    IMAGE_PATH = "./FastSAM/images/imageForSegmentation.png"

    everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=2872, conf=0.4, iou=0.9,)
    prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)


    ann = compute_segment.remote(prompt_process)

    prompt_process.plot(annotations=ann, output_path="./FastSAM/output/dishes_annotated.png")

    print("process completed")

if __name__ == "__main__":
    main()