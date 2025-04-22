import os
import torch
import torch.nn as nn


def save_model(path, model, current_epoch, dataset_name, flag='img'):
    os.makedirs(path, exist_ok=True)
    state_dict = {'epoch': current_epoch,
                  'nfs': [nf.state_dict() for nf in model.nfs],
                  'phi_inters': model.phi_inters.state_dict(),
                  'phi_intras': model.phi_intras.state_dict(),
                  'mus': model.mus.state_dict(),
                  'mu_deltas': model.mu_deltas.state_dict()}
    torch.save(state_dict, os.path.join(path, f'HGAD_{dataset_name}_{flag}.pt'))
    print('Saving model weights into {}'.format(os.path.join(path, f'HGAD_{dataset_name}_{flag}.pt')))


def load_model(path, model):
    state_dict = torch.load(path)
    # model.nfs = [nf.load_state_dict(state, strict=False) for nf, state in zip(model.nfs, state_dict['nfs'])]
    for nf, state in zip(model.nfs, state_dict['nfs']):
        nf.load_state_dict(state, strict=False)
    model.phi_inters.load_state_dict(state_dict['phi_inters'])
    model.phi_intras.load_state_dict(state_dict['phi_intras'])
    model.mus.load_state_dict(state_dict['mus'])
    model.mu_deltas.load_state_dict(state_dict['mu_deltas'])
    print('Loading model weights from {}'.format(path))


def load_model_partial(path, model, curr_num_classes):
    state_dict = torch.load(path)

    # === 자동으로 prev_num_classes 추론 ===
    example_key = next(iter(state_dict['phi_inters']))
    prev_num_classes = state_dict['phi_inters'][example_key].shape[0]
    
    if curr_num_classes < prev_num_classes:
        raise ValueError(f"[ERROR] curr_num_classes ({curr_num_classes}) < prev_num_classes ({prev_num_classes})")
    
    print(f"Detected previous class count: {prev_num_classes}")

    # === Normalizing Flows는 그대로 로드
    for nf, state in zip(model.nfs, state_dict['nfs']):
        nf.load_state_dict(state, strict=False)

    # === phi_inters와 phi_intras 확장
    def expand_paramlist(plist_key, shape_type="vector"):
        orig_params_dict = state_dict[plist_key]
        new_list = nn.ParameterList()
        for i in range(len(orig_params_dict)):
            orig = orig_params_dict[str(i)]

            if shape_type == "vector":
                new_param = nn.Parameter(torch.zeros(curr_num_classes, device=orig.device))
                new_param.data[:prev_num_classes] = orig[:prev_num_classes]

            elif shape_type == "matrix":
                # shape: (n_classes, n_centers)
                new_param = nn.Parameter(torch.zeros(curr_num_classes, orig.shape[1], device=orig.device))
                new_param.data[:prev_num_classes] = orig[:prev_num_classes]

            else:
                raise ValueError("Unsupported shape_type for expand_paramlist")

            new_list.append(new_param)
        return new_list

    model.phi_inters = expand_paramlist('phi_inters', shape_type="vector")
    model.phi_intras = expand_paramlist('phi_intras', shape_type="matrix")


    # === mus
    for i in range(len(model.mus)):
        old_mu = state_dict['mus'][str(i)]
        new_mu = torch.zeros((curr_num_classes, old_mu.shape[1]), device=old_mu.device)
        new_mu[:prev_num_classes] = old_mu[:prev_num_classes]
        model.mus[i] = nn.Parameter(new_mu)

    # === mu_deltas
    for i in range(len(model.mu_deltas)):
        old_d = state_dict['mu_deltas'][str(i)]
        new_d = torch.zeros((curr_num_classes, old_d.shape[1], old_d.shape[2]), device=old_d.device)
        new_d[:prev_num_classes] = old_d[:prev_num_classes]
        model.mu_deltas[i] = nn.Parameter(new_d)

    print(f"Partially loaded weights from {path} with automatic class expansion: {prev_num_classes} → {curr_num_classes}")