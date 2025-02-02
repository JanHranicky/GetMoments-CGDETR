import torch

from run_on_video.data_utils import ClipFeatureExtractor
from run_on_video.model_utils import build_inference_model
from utils.tensor_utils import pad_sequences_1d
from cg_detr.span_utils import span_cxw_to_xx
import torch.nn.functional as F
import json

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

class CGDETRPredictor:
    def __init__(self, ckpt_path, clip_model_name_or_path="ViT-B/32", device="cuda"):
        self.clip_len = 2  # seconds
        self.device = device
        print("Loading feature extractors...")
        self.feature_extractor = ClipFeatureExtractor(
            framerate=1/self.clip_len, size=224, centercrop=True,
            model_name_or_path=clip_model_name_or_path, device=device
        )
        print("Loading trained CG-DETR model...")
        self.model = build_inference_model(ckpt_path).to(self.device)

    @torch.no_grad()
    def localize_moment(self, video_path, query_list, mode):
        """
        Args:
            video_path: str, path to the video file
            query_list: List[str], each str is a query for this video
        """
        # construct model inputs
        n_query = len(query_list)
        video_feats = self.feature_extractor.encode_video(video_path)
        video_feats = F.normalize(video_feats, dim=-1, eps=1e-5)
        n_frames = len(video_feats)
        # add tef
        tef_st = torch.arange(0, n_frames, 1.0) / n_frames
        tef_ed = tef_st + 1.0 / n_frames
        tef = torch.stack([tef_st, tef_ed], dim=1).to(self.device)  # (n_frames, 2)
        video_feats = torch.cat([video_feats, tef], dim=1)
        assert n_frames <= 75, "The positional embedding of this pretrained CGDETR only support video up " \
                               "to 150 secs (i.e., 75 2-sec clips) in length"
        video_feats = video_feats.unsqueeze(0).repeat(n_query, 1, 1)  # (#text, T, d)
        video_mask = torch.ones(n_query, n_frames).to(self.device)
        query_feats = self.feature_extractor.encode_text(query_list)  # #text * (L, d)
        
        if mode is not None: #if mode set to 0/1 make the input query constant 
            for n in range(len(query_feats)):
                query_feats[n].fill_(mode)
        
        query_feats, query_mask = pad_sequences_1d(
            query_feats, dtype=torch.float32, device=self.device, fixed_length=None)
        query_feats = F.normalize(query_feats, dim=-1, eps=1e-5)
        model_inputs = dict(
            src_vid=video_feats,
            src_vid_mask=video_mask,
            src_txt=query_feats,
            src_txt_mask=query_mask,
            vid=None,
            qid=None
        )

        # decode outputs
        outputs = self.model(**model_inputs)
        # #moment_queries refers to the positional embeddings in CGDETR's decoder, not the input text query
        prob = F.softmax(outputs["pred_logits"], -1)  # (batch_size, #moment_queries=10, #classes=2)
        scores = prob[..., 0]  # * (batch_size, #moment_queries)  foreground label is 0, we directly take it
        pred_spans = outputs["pred_spans"]  # (bsz, #moment_queries, 2)
        _saliency_scores = outputs["saliency_scores"].half()  # (bsz, L)
        saliency_scores = []
        valid_vid_lengths = model_inputs["src_vid_mask"].sum(1).cpu().tolist()
        for j in range(len(valid_vid_lengths)):
            _score = _saliency_scores[j, :int(valid_vid_lengths[j])].tolist()
            _score = [round(e, 4) for e in _score]
            saliency_scores.append(_score)

        # compose predictions
        predictions = []
        video_duration = n_frames * self.clip_len
        for idx, (spans, score) in enumerate(zip(pred_spans.cpu(), scores.cpu())):
            spans = span_cxw_to_xx(spans) * video_duration
            # # (#queries, 3), [st(float), ed(float), score(float)]
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            cur_query_pred = dict(
                query=query_list[idx],  # str
                vid=video_path,
                pred_relevant_windows=cur_ranked_preds,  # List([st(float), ed(float), score(float)])
                pred_saliency_scores=saliency_scores[idx]  # List(float), len==n_frames, scores for each frame
            )
            predictions.append(cur_query_pred)

        return predictions

def run_inference(mode):
    # load example data
    from utils.basic_utils import load_jsonl
    
    video_folder_path = "run_on_video/example/"
    videos = ["concert_multi.mp4", "day_multi.mp4", "people_multi.mp4", "selfie_day.mp4", "selfie_night.mp4", "watch.mp4"]
        
    query_path = "run_on_video/example/queries.jsonl"
    queries = load_jsonl(query_path)
    query_text_list = [e["query"] for e in queries]
    ckpt_path = "run_on_video/CLIP_ckpt/qvhighlights_onlyCLIP/model_best.ckpt"

    # run predictions
    print("Build models...")
    clip_model_name_or_path = "ViT-B/32"
    # clip_model_name_or_path = "tmp/ViT-B-32.pt"
    cg_detr_predictor = CGDETRPredictor(
        ckpt_path=ckpt_path,
        clip_model_name_or_path=clip_model_name_or_path,
        device="cuda"
    )
    
    data_dict = {}
    for i,v in enumerate(videos):
        print(f"Running predictions for video {i+1}/{len(videos)}: {v}")
        video_path = video_folder_path + v
        
        predictions = cg_detr_predictor.localize_moment(
            video_path=video_path, query_list=query_text_list, mode=mode)
        data_dict[v] = predictions
        # print data
        query_data = queries[i]
        
        print("-"*30 + f"idx{i}")
        print(f">> query: {query_data['query']}")
        print(f">> video_path: {video_path}")
        print(f">> Predicted moments ([start_in_seconds, end_in_seconds, score]): "
            f"{predictions[i]['pred_relevant_windows']}")
        pred_saliency_scores = torch.Tensor(predictions[i]['pred_saliency_scores'])
        bias = 0 - pred_saliency_scores.min()
        pred_saliency_scores += bias
        print(f">> Most saliency clip is (for all 2-sec clip): "
            f"{pred_saliency_scores.argmax()}")
        print(f">> Predicted saliency scores (for all 2-sec clip): "
            f"{pred_saliency_scores.tolist()}")
        print(len(pred_saliency_scores))
    
    return data_dict

import pickle 

if __name__ == "__main__":
    MODES = [0,1,None]
    MODES2STR = ['zeroes','ones','empty']
    
    for idx,mode in enumerate(MODES):
        predictions = run_inference(mode)
        
        if predictions:
            with open(f'get_moments_predictions_{MODES2STR[idx]}_query.pkl', 'wb') as f:
                pickle.dump(predictions, f)