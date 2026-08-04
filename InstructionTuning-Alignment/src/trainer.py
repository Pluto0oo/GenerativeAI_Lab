import torch
import torch.nn as nn
import torch.optim as optim
import learn2learn as l2l
from tqdm import tqdm
from typing import Dict, List, Tuple
from .data_loader import split_support_query


def train_protonet(model: nn.Module, train_loader, config: Dict, device: torch.device, logger) -> Dict:
    optimizer = optim.Adam(model.parameters(), lr=config['training']['meta_lr'], 
                          weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    
    criterion = nn.CrossEntropyLoss()
    train_shots = config['data']['train_shots']
    
    metrics_history = []
    
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_tasks = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}", leave=False) as pbar:
            for batch in pbar:
                optimizer.zero_grad()
                
                support_images, support_labels, query_images, query_labels = split_support_query(batch, train_shots)
                support_images, support_labels = support_images.to(device), support_labels.to(device)
                query_images, query_labels = query_images.to(device), query_labels.to(device)
                
                logits = model(support_images, support_labels, query_images)
                loss = criterion(logits, query_labels)
                loss.backward()
                
                if config['training']['clip_grad_norm'] > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), config['training']['clip_grad_norm'])
                
                optimizer.step()
                
                acc = (logits.argmax(dim=1) == query_labels).float().mean().item()
                total_loss += loss.item() * query_images.size(0)
                total_acc += acc * query_images.size(0)
                num_tasks += query_images.size(0)
                
                pbar.set_postfix({'loss': loss.item(), 'acc': acc})
        
        avg_loss = total_loss / num_tasks
        avg_acc = total_acc / num_tasks
        scheduler.step()
        
        metrics_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'accuracy': avg_acc,
        })
        
        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={avg_acc:.4f}")
    
    return metrics_history


def train_finetune(model: nn.Module, train_loader, config: Dict, device: torch.device, logger) -> Dict:
    maml = l2l.algorithms.MAML(model, lr=config['training']['fast_lr'], first_order=True)
    meta_optimizer = optim.Adam(maml.parameters(), lr=config['training']['meta_lr'],
                               weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(meta_optimizer, T_max=config['training']['epochs'])
    
    criterion = nn.CrossEntropyLoss()
    train_shots = config['data']['train_shots']
    inner_steps = config['training']['inner_steps']
    
    metrics_history = []
    
    for epoch in range(config['training']['epochs']):
        maml.train()
        total_loss = 0.0
        total_acc = 0.0
        num_tasks = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}", leave=False) as pbar:
            for batch in pbar:
                meta_optimizer.zero_grad()
                
                support_images, support_labels, query_images, query_labels = split_support_query(batch, train_shots)
                support_images, support_labels = support_images.to(device), support_labels.to(device)
                query_images, query_labels = query_images.to(device), query_labels.to(device)
                
                learner = maml.clone()
                
                for _ in range(inner_steps):
                    support_logits = learner(support_images)
                    inner_loss = criterion(support_logits, support_labels)
                    learner.adapt(inner_loss)
                
                query_logits = learner(query_images)
                meta_loss = criterion(query_logits, query_labels)
                meta_loss.backward()
                
                if config['training']['clip_grad_norm'] > 0:
                    nn.utils.clip_grad_norm_(maml.parameters(), config['training']['clip_grad_norm'])
                
                meta_optimizer.step()
                
                acc = (query_logits.argmax(dim=1) == query_labels).float().mean().item()
                total_loss += meta_loss.item() * query_images.size(0)
                total_acc += acc * query_images.size(0)
                num_tasks += query_images.size(0)
                
                pbar.set_postfix({'loss': meta_loss.item(), 'acc': acc})
        
        avg_loss = total_loss / num_tasks
        avg_acc = total_acc / num_tasks
        scheduler.step()
        
        metrics_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'accuracy': avg_acc,
        })
        
        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={avg_acc:.4f}")
    
    maml.module.to(device)
    return metrics_history


def train_maml(model: nn.Module, train_loader, config: Dict, device: torch.device, logger) -> Dict:
    maml = l2l.algorithms.MAML(model, lr=config['training']['fast_lr'], first_order=False)
    optimizer = optim.Adam(maml.parameters(), lr=config['training']['meta_lr'],
                          weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    
    criterion = nn.CrossEntropyLoss()
    train_shots = config['data']['train_shots']
    train_ways = config['data']['train_ways']
    inner_steps = config['training']['inner_steps']
    
    metrics_history = []
    
    for epoch in range(config['training']['epochs']):
        maml.train()
        total_loss = 0.0
        total_acc = 0.0
        num_tasks = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}", leave=False) as pbar:
            for batch in pbar:
                optimizer.zero_grad()
                
                support_images, support_labels, query_images, query_labels = split_support_query(batch, train_shots)
                support_images, support_labels = support_images.to(device), support_labels.to(device)
                query_images, query_labels = query_images.to(device), query_labels.to(device)
                
                learner = maml.clone()
                
                for _ in range(inner_steps):
                    support_logits = learner(support_images)
                    inner_loss = criterion(support_logits, support_labels)
                    learner.adapt(inner_loss)
                
                query_logits = learner(query_images)
                meta_loss = criterion(query_logits, query_labels)
                meta_loss.backward()
                
                if config['training']['clip_grad_norm'] > 0:
                    nn.utils.clip_grad_norm_(maml.parameters(), config['training']['clip_grad_norm'])
                
                optimizer.step()
                
                acc = (query_logits.argmax(dim=1) == query_labels).float().mean().item()
                total_loss += meta_loss.item() * query_images.size(0)
                total_acc += acc * query_images.size(0)
                num_tasks += query_images.size(0)
                
                pbar.set_postfix({'loss': meta_loss.item(), 'acc': acc})
        
        avg_loss = total_loss / num_tasks
        avg_acc = total_acc / num_tasks
        scheduler.step()
        
        metrics_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'accuracy': avg_acc,
        })
        
        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={avg_acc:.4f}")
    
    maml.module.to(device)
    return metrics_history


def evaluate_protonet(model: nn.Module, test_loader, config: Dict, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_shots = config['data']['test_shots']
    
    total_loss = 0.0
    total_acc = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            support_images, support_labels, query_images, query_labels = split_support_query(batch, test_shots)
            support_images, support_labels = support_images.to(device), support_labels.to(device)
            query_images, query_labels = query_images.to(device), query_labels.to(device)
            
            logits = model(support_images, support_labels, query_images)
            loss = criterion(logits, query_labels)
            
            acc = (logits.argmax(dim=1) == query_labels).float().mean().item()
            total_loss += loss.item() * query_images.size(0)
            total_acc += acc * query_images.size(0)
            num_samples += query_images.size(0)
    
    avg_loss = total_loss / num_samples
    avg_acc = total_acc / num_samples
    
    return avg_loss, avg_acc


def evaluate_finetune(model: nn.Module, test_loader, config: Dict, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_shots = config['data']['test_shots']
    test_ways = config['data']['test_ways']
    embedding_dim = config['model']['embedding_dim']
    inner_steps = config['training']['inner_steps']
    fast_lr = config['training']['fast_lr']
    
    total_loss = 0.0
    total_acc = 0.0
    num_samples = 0
    
    for batch in test_loader:
        support_images, support_labels, query_images, query_labels = split_support_query(batch, test_shots)
        support_images, support_labels = support_images.to(device), support_labels.to(device)
        query_images, query_labels = query_images.to(device), query_labels.to(device)
        
        model_copy = type(model)(backbone=config['model']['backbone'], 
                                 hidden_size=config['model']['hidden_size'],
                                 embedding_dim=embedding_dim,
                                 num_layers=config['model'].get('num_layers', 4))
        model_copy.load_state_dict(model.state_dict())
        model_copy.set_classifier(test_ways, embedding_dim)
        model_copy = model_copy.to(device)
        
        inner_params = list(model_copy.parameters())
        for param in inner_params:
            param.requires_grad = True
        
        inner_optimizer = optim.SGD(inner_params, lr=fast_lr)
        
        for _ in range(inner_steps):
            inner_optimizer.zero_grad()
            support_logits = model_copy(support_images)
            inner_loss = criterion(support_logits, support_labels)
            inner_loss.backward()
            inner_optimizer.step()
        
        model_copy.eval()
        with torch.no_grad():
            query_logits = model_copy(query_images)
            loss = criterion(query_logits, query_labels)
            
            acc = (query_logits.argmax(dim=1) == query_labels).float().mean().item()
            total_loss += loss.item() * query_images.size(0)
            total_acc += acc * query_images.size(0)
            num_samples += query_images.size(0)
    
    avg_loss = total_loss / num_samples
    avg_acc = total_acc / num_samples
    
    return avg_loss, avg_acc


def evaluate_maml(model: nn.Module, test_loader, config: Dict, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_shots = config['data']['test_shots']
    inner_steps = config['training']['inner_steps']
    fast_lr = config['training']['fast_lr']
    
    total_loss = 0.0
    total_acc = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            support_images, support_labels, query_images, query_labels = split_support_query(batch, test_shots)
            support_images, support_labels = support_images.to(device), support_labels.to(device)
            query_images, query_labels = query_images.to(device), query_labels.to(device)
            
            learner = l2l.clone_module(model)
            
            for _ in range(inner_steps):
                support_logits = learner(support_images)
                inner_loss = criterion(support_logits, support_labels)
                l2l.update_module(learner, inner_loss, lr=fast_lr)
            
            query_logits = learner(query_images)
            loss = criterion(query_logits, query_labels)
            
            acc = (query_logits.argmax(dim=1) == query_labels).float().mean().item()
            total_loss += loss.item() * query_images.size(0)
            total_acc += acc * query_images.size(0)
            num_samples += query_images.size(0)
    
    avg_loss = total_loss / num_samples
    avg_acc = total_acc / num_samples
    
    return avg_loss, avg_acc


def train_model(model: nn.Module, train_loader, config: Dict, device: torch.device, logger) -> Dict:
    model_type = config['model']['type']
    
    if model_type == "protonet":
        return train_protonet(model, train_loader, config, device, logger)
    elif model_type == "finetune":
        return train_finetune(model, train_loader, config, device, logger)
    elif model_type == "maml":
        return train_maml(model, train_loader, config, device, logger)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_model(model: nn.Module, test_loader, config: Dict, device: torch.device) -> Tuple[float, float]:
    model_type = config['model']['type']
    
    if model_type == "protonet":
        return evaluate_protonet(model, test_loader, config, device)
    elif model_type == "finetune":
        return evaluate_finetune(model, test_loader, config, device)
    elif model_type == "maml":
        return evaluate_maml(model, test_loader, config, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
