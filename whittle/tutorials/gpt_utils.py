from datasets import load_dataset
from tqdm import tqdm
import transformers

def evaluate_wikitext(model, tokenizer):
      test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
      encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
      max_length = model.max_seq_length
      seq_len = encodings.input_ids.size(1)
      nlls = []
      device = "cuda:0" if torch.cuda.is_available() else "cpu"
      prev_end_loc = 0
      model.to(device)
      model.eval()
      for begin_loc in tqdm(range(0, seq_len, max_length)):
          end_loc = min(begin_loc + max_length, seq_len - 1)
          trg_len = (
              end_loc - prev_end_loc
          )  # may be different from stride on last loop
          input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
          target_ids = input_ids.clone()
          target_ids = encodings.input_ids[:, begin_loc + 1 : end_loc + 1].to(
              device
          )

          with torch.no_grad():
              outputs = model(input_ids)
              neg_log_likelihood = torch.nn.CrossEntropyLoss()(
                  outputs.view(-1, outputs.size(-1)), target_ids.view(-1)
              )
          nlls.append(neg_log_likelihood)
          prev_end_loc = end_loc
          if end_loc == seq_len:
              break
      model.reset_super_network()
      ppl = torch.exp(torch.stack(nlls).mean())
      return ppl.item()
