import torch
from torch import Tensor, nn

from lerobot.policies.sac.reward_model.modeling_classifier import Classifier, ClassifierOutput
from lerobot.utils.constants import OBS_IMAGE, OBS_STATE, REWARD

from share.policies.reward_classifier.configuration_state_reward_classifier import StateRewardClassifierConfig


class StateRewardClassifier(Classifier):
    """Stock reward classifier with an additional MLP branch for the state vector.

    The per-camera image encoders and training/eval semantics are inherited unchanged;
    only the classifier head is rebuilt to take the concatenated image latents plus the
    encoded state.
    """

    name = "state_reward_classifier"
    config_class = StateRewardClassifierConfig

    def __init__(self, config: StateRewardClassifierConfig):
        super().__init__(config)

        state_dim = config.input_features[OBS_STATE].shape[0]
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, config.state_hidden_dim),
            nn.LayerNorm(config.state_hidden_dim),
            nn.Tanh(),
        )

        # Rebuild the head built by the parent to accept the extra state features.
        input_dim = config.latent_dim if self.is_cnn else self.encoder.config.hidden_size
        self.classifier_head = nn.Sequential(
            nn.Linear(input_dim * config.num_cameras + config.state_hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout_rate),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1 if config.num_classes == 2 else config.num_classes),
        )

    def predict(self, xs: list, state: Tensor | None = None) -> ClassifierOutput:
        if state is None:
            raise ValueError("StateRewardClassifier.predict requires the state tensor.")
        encoder_outputs = torch.hstack(
            [self._get_encoder_output(x, img_key) for x, img_key in zip(xs, self.image_keys, strict=True)]
            + [self.state_encoder(state)]
        )
        logits = self.classifier_head(encoder_outputs)

        if self.config.num_classes == 2:
            logits = logits.squeeze(-1)
            probabilities = torch.sigmoid(logits)
        else:
            probabilities = torch.softmax(logits, dim=-1)

        return ClassifierOutput(logits=logits, probabilities=probabilities, hidden_states=encoder_outputs)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        images = [batch[key] for key in self.config.input_features if key.startswith(OBS_IMAGE)]
        labels = batch[REWARD]
        outputs = self.predict(images, state=batch[OBS_STATE])

        if self.config.num_classes == 2:
            loss = nn.functional.binary_cross_entropy_with_logits(outputs.logits, labels)
            predictions = (torch.sigmoid(outputs.logits) > 0.5).float()
        else:
            loss = nn.functional.cross_entropy(outputs.logits, labels.long())
            predictions = torch.argmax(outputs.logits, dim=1)

        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        return loss, {"accuracy": 100 * correct / total, "correct": correct, "total": total}
